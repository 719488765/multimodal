#!/usr/bin/env python3
"""SDAVT v3 前期落地校验：配置、目录、标签映射、loader 冒烟（不启动训练）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.collate import multimodal_collate_fn
from data.dataset import MultimodalDataset
from models.multimodal_model import MultimodalEmotionModel
from utils.helpers import get_dataloader_kwargs, load_config, setup_device
from utils.label_mapping import get_emotion_class_names, uses_native_labels

SDAVT_CONFIGS = [
    "config/sdavt_v3/meld/S1_M0_AVT_ES_baseline.yaml",
    "config/sdavt_v3/meld/S1_M1_AVT_ES_v2recipe.yaml",
    "config/sdavt_v3/meld/S1_M0N_AVT_ES_native.yaml",
    "config/sdavt_v3/crema/S1_C0_AVT_ES_baseline.yaml",
    "config/sdavt_v3/mosei/S1_O0_AVT_ES_npy.yaml",
]

REQUIRED_DIRS = [
    "config/sdavt_v3/meld",
    "config/sdavt_v3/crema",
    "config/sdavt_v3/mosei",
    "logs_sdavt_v3",
    "checkpoints_sdavt_v3",
    "outputs_sdavt_v3",
]

REQUIRED_TRAINING_KEYS = (
    "early_stopping",
    "save_every_n_epochs",
    "seed",
    "learning_rate",
    "weight_decay",
)


class PrepValidationError(Exception):
    pass


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise PrepValidationError(msg)


def check_directories() -> None:
    print("\n==> Directory structure")
    for rel in REQUIRED_DIRS:
        path = ROOT / rel
        if path.is_dir():
            _ok(rel)
        else:
            _fail(f"missing directory: {rel}")


def check_yaml(config_rel: str) -> dict:
    print(f"\n==> YAML lint: {config_rel}")
    path = ROOT / config_rel
    if not path.is_file():
        _fail(f"missing config: {config_rel}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for section in ("model", "training", "datasets", "data", "paths", "experiment"):
        if section not in cfg:
            _fail(f"{config_rel}: missing top-level '{section}'")

    paths = cfg["paths"]
    for key in ("checkpoint_dir", "log_dir", "output_dir"):
        val = paths.get(key, "")
        if "sdavt_v3" not in str(val):
            _fail(f"{config_rel}: paths.{key} must contain 'sdavt_v3', got {val!r}")
        _ok(f"paths.{key} = {val}")

    es = cfg["training"].get("early_stopping", {})
    if not es.get("enabled", False):
        _fail(f"{config_rel}: early_stopping.enabled must be true")
    _ok("early_stopping.enabled = true")

    if cfg["training"].get("save_every_n_epochs", 5) != 0:
        _fail(f"{config_rel}: save_every_n_epochs should be 0 (best-only)")
    _ok("save_every_n_epochs = 0")

    for key in REQUIRED_TRAINING_KEYS:
        if key not in cfg["training"]:
            _fail(f"{config_rel}: training.{key} missing")

    ds_list = cfg["training"].get("pretrain", {}).get("datasets") or []
    if len(ds_list) != 1:
        _fail(f"{config_rel}: pretrain.datasets must have exactly one entry")
    _ok(f"single-domain pretrain: {ds_list[0]}")

    if cfg.get("model", {}).get("domain_adaptation", {}).get("enabled", False):
        _fail(f"{config_rel}: domain_adaptation must be disabled")
    _ok("domain_adaptation disabled")

    exp = cfg.get("experiment", {})
    if exp.get("project") != "sdavt_v3":
        _fail(f"{config_rel}: experiment.project should be sdavt_v3")
    _ok(f"experiment.name = {exp.get('name')}")

    return cfg


def _subset_indices(base_ds, ds_name: str) -> list[int]:
    ds_id = {"crema": 0, "meld": 1, "mosei": 2}[ds_name]
    return [
        i
        for i, s in enumerate(base_ds.data_list)
        if s.get("dataset_id", -1) == ds_id
    ]


def smoke_loader(config_rel: str, cfg: dict, batch_size: int = 2) -> None:
    print(f"\n==> Loader smoke: {config_rel}")
    ds_name = str(cfg["training"]["pretrain"]["datasets"][0]).lower()
    data_root = cfg["data"]["root_dir"]
    if not (ROOT / data_root).is_dir():
        print(f"  [SKIP] data root not found: {data_root}")
        return

    base_ds = MultimodalDataset(data_root, split="val", config=cfg)
    indices = _subset_indices(base_ds, ds_name)
    if not indices:
        _fail(f"{config_rel}: no samples for dataset {ds_name}")
    _ok(f"found {len(indices)} val samples for {ds_name}")

    subset = torch.utils.data.Subset(base_ds, indices[: min(4, len(indices))])
    dl_kwargs = get_dataloader_kwargs(cfg, shuffle=False)
    dl_kwargs["collate_fn"] = multimodal_collate_fn
    loader = torch.utils.data.DataLoader(subset, batch_size=min(batch_size, len(subset)), **dl_kwargs)
    batch = next(iter(loader))

    if batch["video"] is None:
        _fail(f"{config_rel}: video batch is None")
    v = batch["video"]
    _ok(f"video shape {tuple(v.shape)}")

    if batch.get("audio") is None and batch.get("audio_precomputed") is None:
        _fail(f"{config_rel}: no audio / audio_precomputed")
    if batch.get("audio_precomputed") is not None:
        _ok(f"audio_precomputed shape {tuple(batch['audio_precomputed'].shape)}")
    elif batch.get("audio") is not None:
        _ok(f"audio shape {tuple(batch['audio'].shape)}")

    if batch["text_input_ids"] is None:
        _fail(f"{config_rel}: text batch is None")
    _ok(f"text batch {tuple(batch['text_input_ids'].shape)}")

    n_cls = cfg["model"]["output"]["emotion_classes"]
    labels = batch["emotion_label"]
    if labels.max().item() >= n_cls or labels.min().item() < 0:
        _fail(f"{config_rel}: label out of range [0,{n_cls})")
    _ok(f"labels in [0,{n_cls}), native={uses_native_labels(ds_name, cfg.get('datasets', {}))}")


def smoke_forward(config_rel: str, cfg: dict) -> None:
    print(f"\n==> Forward smoke: {config_rel}")
    ds_name = str(cfg["training"]["pretrain"]["datasets"][0]).lower()
    data_root = cfg["data"]["root_dir"]
    if not (ROOT / data_root).is_dir():
        print(f"  [SKIP] data root not found")
        return

    device = setup_device(cfg)
    base_ds = MultimodalDataset(data_root, split="val", config=cfg)
    indices = _subset_indices(base_ds, ds_name)
    subset = torch.utils.data.Subset(base_ds, indices[:2])
    dl_kwargs = get_dataloader_kwargs(cfg, shuffle=False)
    dl_kwargs["collate_fn"] = multimodal_collate_fn
    loader = torch.utils.data.DataLoader(subset, batch_size=min(2, len(subset)), **dl_kwargs)
    batch = next(iter(loader))

    model = MultimodalEmotionModel(cfg).to(device)
    model.eval()
    with torch.no_grad():
        inputs = {
            "video": batch["video"].to(device) if batch.get("video") is not None else None,
            "audio": batch["audio"].to(device) if batch.get("audio") is not None else None,
            "audio_precomputed": batch["audio_precomputed"].to(device)
            if batch.get("audio_precomputed") is not None
            else None,
            "physiological": batch["physiological"].to(device),
            "text_input_ids": batch["text_input_ids"].to(device),
            "text_attention_mask": batch["text_attention_mask"].to(device),
        }
        out = model(**inputs, return_domain_logits=False)
        logits = out["emotion_logits"]
        if torch.isnan(logits).any():
            _fail(f"{config_rel}: NaN in emotion_logits")
        n_cls = cfg["model"]["output"]["emotion_classes"]
        if logits.shape[-1] != n_cls:
            _fail(f"{config_rel}: logits dim {logits.shape[-1]} != emotion_classes {n_cls}")
    _ok(f"forward OK, logits {tuple(logits.shape)}")


def check_native_label_mapping() -> None:
    print("\n==> Native label mapping")
    from utils.label_mapping import NATIVE_EMOTION_MAPS, get_sample_emotion_map

    for ds in ("meld", "crema", "mosei"):
        m = NATIVE_EMOTION_MAPS[ds]
        n = max(m.values()) + 1
        _ok(f"{ds}: {n} native classes, keys={len(m)}")
        cfg = {"datasets": {ds: {"use_native_labels": True}}}
        assert get_sample_emotion_map(ds, cfg["datasets"]) == m


def check_modality_counts() -> None:
    print("\n==> Modality completeness (optional)")
    try:
        from scripts.transcribe_crema_text import count_crema_text
        from scripts.extract_mosei_text_from_sdk import count_mosei_text

        crema = count_crema_text(str(ROOT / "data"), ("train", "val", "test"))
        mosei = count_mosei_text(str(ROOT / "data"), ("train", "val", "test"))
        _ok(f"CREMA text counts {crema}")
        _ok(f"MOSEI text counts {mosei}")
    except Exception as exc:
        print(f"  [SKIP] modality count scripts: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SDAVT v3 prep (no training)")
    parser.add_argument("--skip-forward", action="store_true", help="Skip GPU/CPU forward pass")
    parser.add_argument("--skip-loader", action="store_true", help="Skip dataloader smoke")
    args = parser.parse_args()

    print("SDAVT v3 prep validation")
    try:
        check_directories()
        check_native_label_mapping()
        configs = []
        for rel in SDAVT_CONFIGS:
            configs.append((rel, check_yaml(rel)))

        if not args.skip_loader:
            for rel, cfg in configs:
                smoke_loader(rel, cfg)

        if not args.skip_forward:
            for rel, cfg in configs:
                smoke_forward(rel, cfg)

        check_modality_counts()
    except PrepValidationError as exc:
        print(f"\nValidation FAILED: {exc}")
        return 1

    print("\n=== All SDAVT v3 prep checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

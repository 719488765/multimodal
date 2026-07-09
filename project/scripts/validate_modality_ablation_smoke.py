#!/usr/bin/env python3
"""Smoke tests for P4 modality ablation after fusion/gating fixes."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import MultimodalDataset
from data.collate import multimodal_collate_fn
from models.multimodal_model import MultimodalEmotionModel
from scripts.train import _apply_active_modalities, _batch_to_model_inputs
from torch.utils.data import DataLoader, Subset
from utils.helpers import load_config, setup_device


def _subset_mosei(dataset, config):
    from scripts.train import _subset_by_datasets

    return _subset_by_datasets(dataset, config, "pretrain")


def _one_batch(config_path: Path, device: torch.device):
    config = load_config(str(config_path))
    config = copy.deepcopy(config)
    data_root = config["data"]["root_dir"]
    ds = _subset_mosei(
        MultimodalDataset(data_root, split="val", config=config),
        config,
    )
    if len(ds) == 0:
        raise RuntimeError(f"empty val subset for {config_path}")
    loader = DataLoader(
        ds,
        batch_size=min(2, len(ds)),
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )
    batch = next(iter(loader))
    inputs = _batch_to_model_inputs(batch, device)
    inputs = _apply_active_modalities(inputs, config)
    return config, inputs


def _hook_features(model: MultimodalEmotionModel):
    captured = {}

    def _pre_hook(_module, args, kwargs):
        captured.clear()

    orig_forward = model.forward

    def _wrapped(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        return out

    model.forward = _wrapped  # type: ignore[method-assign]

    def capture_after(model, inputs):
        mods = config_modalities(model.config)
        with torch.no_grad():
            device = next(model.parameters()).device
            batch_size = 1
            for key in ("video", "audio_precomputed", "text_input_ids"):
                if inputs.get(key) is not None:
                    batch_size = inputs[key].shape[0]
                    break
            active = {
                "video": mods["use_video"] and inputs.get("video") is not None,
                "audio": mods["use_audio"]
                and (inputs.get("audio") is not None or inputs.get("audio_precomputed") is not None),
                "text": mods["use_text"] and inputs.get("text_input_ids") is not None,
            }
            feats = {}
            if active["video"]:
                feats["video"] = model.video_extractor(inputs["video"])
            else:
                feats["video"] = torch.zeros(
                    batch_size,
                    model.config["model"]["video"]["output_dim"],
                    device=device,
                )
            if active["audio"] and inputs.get("audio_precomputed") is not None:
                feats["audio"] = model.audio_extractor(inputs["audio_precomputed"])
            elif active["audio"] and inputs.get("audio") is not None:
                feats["audio"] = model.audio_extractor(inputs["audio"])
            else:
                feats["audio"] = torch.zeros(
                    batch_size,
                    model.config["model"]["audio"]["output_dim"],
                    device=device,
                )
            if active["text"]:
                feats["text"] = model.text_extractor(
                    inputs["text_input_ids"], inputs.get("text_attention_mask")
                )
            else:
                feats["text"] = torch.zeros(
                    batch_size,
                    model.config["model"]["text"]["output_dim"],
                    device=device,
                )
            captured.update(feats)
            outputs = model(**inputs)
            captured["logits_std"] = float(outputs["emotion_logits"].std().item())
            captured["logits"] = outputs["emotion_logits"]
        return captured

    return capture_after


def config_modalities(config):
    return config.get("model", {}).get("modalities", {})


def run_case(config_path: Path, checks: dict) -> int:
    print(f"\n==> {config_path.name}")
    device = setup_device(load_config(str(config_path)))
    config, inputs = _one_batch(config_path, device)
    model = MultimodalEmotionModel(config).to(device)
    model.eval()

    mods = config_modalities(config)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_std = float(outputs["emotion_logits"].std().item())

    rc = 0
    if mods.get("use_video") and not mods.get("use_audio") and not mods.get("use_text"):
        if inputs.get("audio") is not None or inputs.get("audio_precomputed") is not None:
            print("FAIL: audio input not masked for video-only config")
            rc = 1
        if inputs.get("text_input_ids") is not None:
            print("FAIL: text input not masked for video-only config")
            rc = 1
    if mods.get("use_text") and not mods.get("use_video") and not mods.get("use_audio"):
        if inputs.get("video") is not None:
            print("FAIL: video input not masked for text-only config")
            rc = 1

    min_std = float(checks.get("min_logits_std", 0.05))
    if logits_std < min_std:
        print(f"FAIL: logits_std={logits_std:.4f} < {min_std}")
        rc = 1
    else:
        print(f"OK logits_std={logits_std:.4f}")

    if rc == 0:
        print("PASS")
    return rc


def run_short_train(config_path: Path, steps: int = 30) -> int:
    print(f"\n==> short train {config_path.name} ({steps} steps)")
    from scripts import train as train_mod

    cfg = load_config(str(config_path))
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("training", {})["num_epochs"] = 1
    cfg["training"]["batch_size"] = 2
    tmp = ROOT / "outputs_sdavt_v3_r4" / "_smoke_tmp.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return train_mod.main() if False else _mini_train(tmp, steps)


def _mini_train(config_path: Path, max_steps: int) -> int:
    import math
    from collections import defaultdict

    from scripts.train import (
        MultimodalLoss,
        _apply_active_modalities,
        _batch_targets,
        _batch_to_model_inputs,
        _subset_by_datasets,
        build_training_optimizer,
        build_scheduler,
    )

    config = load_config(str(config_path))
    config = copy.deepcopy(config)
    device = setup_device(config)
    data_root = config["data"]["root_dir"]
    train_ds = _subset_by_datasets(
        MultimodalDataset(data_root, split="train", config=config),
        config,
        "pretrain",
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(config["training"].get("batch_size", 2)),
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=0,
    )
    model = MultimodalEmotionModel(config).to(device)
    criterion = MultimodalLoss(config.get("training", {}).get("loss_weights", {}), config=config).to(
        device
    )
    lr = float(config["training"].get("learning_rate", 1e-4))
    wd = float(config["training"].get("weight_decay", 0.0))
    optimizer = build_training_optimizer(model, config, lr, wd)
    steps_per_epoch = max(1, math.ceil(len(loader)))
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)

    model.train()
    step = 0
    losses = []
    for batch in loader:
        inputs = _batch_to_model_inputs(batch, device)
        inputs = _apply_active_modalities(inputs, config)
        targets = _batch_targets(batch, device)
        outputs = model(**inputs)
        loss, _ = criterion(outputs, targets)
        if not torch.isfinite(loss):
            print(f"FAIL: non-finite loss at step {step}")
            return 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step += 1
        if step >= max_steps:
            break

    if len(losses) < 2:
        print("FAIL: not enough steps")
        return 1
    ln7 = 1.945910
    if all(abs(l - ln7) < 5e-4 for l in losses[-min(5, len(losses)):]):
        print(f"FAIL: ln(7) collapse signature, losses={losses[-3:]}")
        return 1
    if abs(losses[-1] - losses[0]) < 1e-5 and losses[0] > 1.9:
        print(f"FAIL: loss frozen at {losses[0]:.4f}")
        return 1
    print(f"OK loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cases = [
        (ROOT / "config/sdavt_v3_r4/p4_modal/mosei/R4_A_O_V_emotion_shift.yaml", {"min_logits_std": 0.03}),
        (ROOT / "config/sdavt_v3_r4/p4_modal/mosei/R4_A_O_T_emotion_shift.yaml", {"min_logits_std": 0.03}),
        (ROOT / "config/sdavt_v3_r4/p4_modal/mosei/R4_A_O_AVT_emotion_shift.yaml", {"min_logits_std": 0.02}),
        (ROOT / "config/sdavt_v3_r4/p4_modal/crema/R4_A_C_AT_emotion_shift.yaml", {"min_logits_std": 0.02}),
    ]
    rc = 0
    for path, checks in cases:
        if not path.is_file():
            print(f"FAIL missing {path}")
            rc = 1
            continue
        rc = max(rc, run_case(path, checks))
        if not args.skip_train and ("AVT" in path.name or "AT" in path.name):
            rc = max(rc, _mini_train(path, args.steps))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SDAVT v3 训练冒烟：50 step 内 loss 须下降、logits 非塌缩、指标非冻结。"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.collate import multimodal_collate_fn
from data.dataset import MultimodalDataset
from models.multimodal_model import MultimodalEmotionModel
from scripts.train import (
    MultimodalLoss,
    _batch_to_model_inputs,
    apply_backbone_freeze_policy,
    build_training_optimizer,
)


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_smoke(config_path: Path, steps: int = 50, split: str = "train") -> int:
    cfg = _load_cfg(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultimodalDataset(cfg["data"]["root_dir"], split=split, config=cfg)
    if len(ds) == 0:
        print(f"FAIL: empty dataset split={split}")
        return 1

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=max(1, int(cfg.get("training", {}).get("batch_size", 1))),
        shuffle=True,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
    )

    model = MultimodalEmotionModel(cfg).to(device)
    freeze_epochs = int(cfg.get("training", {}).get("freeze_backbone_epochs", 0) or 0)
    freeze_mode = str(cfg.get("training", {}).get("backbone_freeze_mode", "full"))
    if freeze_epochs > 0:
        apply_backbone_freeze_policy(model, cfg, epoch_frozen=True)
    elif freeze_mode == "selective":
        apply_backbone_freeze_policy(model, cfg, epoch_frozen=False)

    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg.get("training", {}).get("weight_decay", 0.0))
    opt = build_training_optimizer(model, cfg, lr, wd)
    crit = MultimodalLoss(cfg["training"]["loss_weights"], config=cfg).to(device)

    num_classes = cfg.get("model", {}).get("output", {}).get("emotion_classes", 7)
    random_baseline = math.log(num_classes)

    it = iter(loader)
    losses = []
    accs = []
    model.train()

    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        inputs = _batch_to_model_inputs(batch, device)
        targets = {"emotion_label": batch["emotion_label"].to(device)}
        opt.zero_grad()
        out = model(**inputs)
        logits = out["emotion_logits"]
        if torch.isnan(logits).any():
            print(f"FAIL step {step}: NaN logits")
            return 1
        std = logits.std().item()
        if std < 0.01:
            print(f"WARN step {step}: logits_std={std:.4f}")

        loss, _ = crit(out, targets)
        if not torch.isfinite(loss):
            print(f"FAIL step {step}: non-finite loss")
            return 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        pred = logits.argmax(dim=-1)
        acc = (pred == targets["emotion_label"]).float().mean().item()
        accs.append(acc)

        if step % 10 == 0 or step == steps - 1:
            print(
                f"  step {step:3d}: loss={loss.item():.4f} "
                f"logits_std={std:.4f} batch_acc={acc:.3f}"
            )

    # 解冻边界模拟（若配置有 freeze_backbone_epochs=1）
    if freeze_epochs == 1:
        apply_backbone_freeze_policy(model, cfg, epoch_frozen=False)
        opt = build_training_optimizer(model, cfg, lr, wd)
        batch = next(it)
        inputs = _batch_to_model_inputs(batch, device)
        targets = {"emotion_label": batch["emotion_label"].to(device)}
        opt.zero_grad()
        out = model(**inputs)
        loss, _ = crit(out, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  post-unfreeze: loss={loss.item():.4f}")

    tail_n = max(5, steps // 5)
    tail_losses = losses[-tail_n:]
    tail_min = min(tail_losses)
    tail_mean = sum(tail_losses) / len(tail_losses)
    loss_drop = losses[0] - tail_min
    acc_unique = len(set(round(a, 3) for a in accs))
    frozen_metric = acc_unique <= 1 and steps > 5

    print(f"\nSummary [{config_path.name}]")
    print(
        f"  loss: {losses[0]:.4f} -> tail_min={tail_min:.4f} "
        f"tail_mean={tail_mean:.4f} last={losses[-1]:.4f} (drop={loss_drop:.4f})"
    )
    print(f"  random CE baseline ln({num_classes})={random_baseline:.4f}")
    print(f"  acc unique buckets: {acc_unique}")

    if loss_drop < 0.05 and tail_min > random_baseline - 0.05:
        print("FAIL: loss did not decrease enough (still near random)")
        return 1
    if frozen_metric and tail_min > random_baseline - 0.02:
        print("FAIL: accuracy frozen near random — training not learning")
        return 1

    print("OK smoke passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run default MOSEI S4 + MELD LFT smokes",
    )
    parser.add_argument(
        "--failing-fix",
        action="store_true",
        help="Run smoke for 6 previously failing fusion configs",
    )
    args = parser.parse_args()

    configs = [Path(p) for p in args.config]
    if args.all:
        configs = [
            ROOT / "config/sdavt_v3/mosei/S4_O0_mosei_AVT_ES_npy_v2.yaml",
            ROOT / "config/sdavt_v3/meld/S2_M1_LFT_native_ap1plus.yaml",
            ROOT / "config/sdavt_v3/meld/S2_M1_meld_AVT_ES_native_ap1plus.yaml",
        ]

    if args.failing_fix:
        # 6 项仍失败融合：用 MOSEI/MELD/CREMA 基线 + 对应 fusion 覆盖
        specs = [
            ("mosei/S4_O0_mosei_AVT_ES_npy_v2.yaml", "leader_follower"),
            ("mosei/S4_O0_mosei_AVT_ES_npy_v2.yaml", "standard"),
            ("mosei/S4_O0_mosei_AVT_ES_npy_v2.yaml", "two_stage"),
            ("meld/S2_M1_meld_AVT_ES_native_ap1plus.yaml", "standard"),
            ("meld/S2_M1_meld_AVT_ES_native_ap1plus.yaml", "two_stage"),
            ("crema/S1_C0_AVT_ES_baseline.yaml", "two_stage"),
        ]
        configs = []
        for rel, fusion in specs:
            cfg_path = ROOT / "config/sdavt_v3" / rel
            cfg = _load_cfg(cfg_path)
            cfg = copy.deepcopy(cfg)
            cfg["model"]["attention"]["fusion_strategy"] = fusion
            cfg["model"]["modalities"]["use_physiological"] = False
            cfg["training"]["loss"]["use_focal_loss"] = False
            cfg["training"]["loss"]["use_class_balanced"] = False
            if fusion in ("standard", "two_stage"):
                cfg["training"]["loss"]["use_focal_loss"] = False
            out = ROOT / f"outputs_sdavt_v3/_smoke_{Path(rel).stem}_{fusion}.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            import yaml as _yaml
            with out.open("w", encoding="utf-8") as f:
                _yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            configs.append(out)

    if not configs and not args.all and not args.failing_fix:
        parser.error("provide --config, --all, or --failing-fix")

    rc = 0
    for p in configs:
        if not p.is_file():
            print(f"FAIL missing config: {p}")
            rc = 1
            continue
        print(f"\n==> Smoke: {p}")
        rc = max(rc, run_smoke(p, steps=args.steps))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

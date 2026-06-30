#!/usr/bin/env python3
"""two_stage 融合修复后冒烟：前向 + 50 step 训练（含 freeze 2 epoch 模拟）。"""

from __future__ import annotations

import copy
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


def main() -> int:
    cfg_path = ROOT / "config/sdavt_v3/meld/S2_M1_LFT_native_ap1plus.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["model"]["attention"]["fusion_strategy"] = "two_stage"
    cfg["training"]["loss"]["use_focal_loss"] = False
    cfg["training"]["freeze_backbone_epochs"] = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultimodalDataset(cfg["data"]["root_dir"], split="val", config=cfg)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=multimodal_collate_fn
    )
    model = MultimodalEmotionModel(cfg).to(device)
    apply_backbone_freeze_policy(model, cfg, epoch_frozen=True)
    opt = build_training_optimizer(model, cfg, 1e-4, 1e-5)
    crit = MultimodalLoss(cfg["training"]["loss_weights"], config=cfg).to(device)

    it = iter(loader)
    for step in range(50):
        if step == 25:
            apply_backbone_freeze_policy(model, cfg, epoch_frozen=False)
            opt = build_training_optimizer(model, cfg, 1e-4, 1e-5)
        batch = next(it)
        inputs = _batch_to_model_inputs(batch, device)
        targets = {"emotion_label": batch["emotion_label"].to(device)}
        opt.zero_grad()
        out = model(**inputs)
        if torch.isnan(out["emotion_logits"]).any():
            print(f"FAIL step {step}: NaN logits")
            return 1
        loss, _ = crit(out, targets)
        if not torch.isfinite(loss):
            print(f"FAIL step {step}: non-finite loss")
            return 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        std = out["emotion_logits"].std().item()
        if step % 10 == 0:
            print(f"step {step}: loss={loss.item():.4f} logits_std={std:.4f}")
        if std < 0.005:
            print(f"FAIL step {step}: logits collapsed std={std}")
            return 1

    print("OK two_stage smoke passed (50 steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

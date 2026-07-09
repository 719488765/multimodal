#!/usr/bin/env python3
"""Bootstrap sdavt_meld_zh_agent checkpoint from M3_M7_combo (skip text_encoder)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models import MultimodalEmotionModel
from utils import load_config, load_checkpoint_partial, remap_legacy_checkpoint_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml",
    )
    parser.add_argument(
        "--resume",
        default="checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth",
    )
    parser.add_argument(
        "--out-dir",
        default="checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent",
    )
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    resume_path = ROOT / args.resume
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    config = load_config(str(cfg_path))
    model = MultimodalEmotionModel(config)

    epoch, loss, loaded, skipped = load_checkpoint_partial(
        str(resume_path),
        model,
        skip_prefixes=["text_extractor."],
        strict=False,
    )
    print(f"Partial load: loaded={loaded} skipped={len(skipped)} epoch={epoch}")

    ckpt = torch.load(resume_path, map_location="cpu")
    out_ckpt = out_dir / "checkpoint_finetune_best_f1.pth"
    payload = {
        "epoch": ckpt.get("epoch", epoch),
        "loss": ckpt.get("loss", loss),
        "model_state_dict": model.state_dict(),
        "bootstrap_from": str(resume_path),
        "note": "bert-base-chinese text_encoder freshly initialized; run finetune for full zh tuning",
    }
    torch.save(payload, out_ckpt)
    print(f"[OK] wrote {out_ckpt} ({out_ckpt.stat().st_size / 1e9:.2f} GB)")

    # convenience copy for pretrain naming if referenced elsewhere
    pretrain_link = out_dir / "checkpoint_pretrain_best_f1.pth"
    if not pretrain_link.exists():
        shutil.copy2(out_ckpt, pretrain_link)
        print(f"[OK] copied -> {pretrain_link.name}")


if __name__ == "__main__":
    main()

"""E4：推理期受控模态缺失实验。

同一份检查点，只改变"缺失时是否把该模态从融合支撑集中排除"（modality_mask），
比较各融合结构在人为缺失下的退化。

用法：
  python scripts/eval_missing_modality.py \
      --config config/e2/f4_es_noseq_meld.yaml \
      --checkpoint checkpoints_e2/xxx/checkpoint_finetune_best_f1.pth \
      --name ESF-zerofill --rates 0,0.15,0.3,0.5 --seed 2024
  # 同一条命令加 --use-mask 即 ESF-mask 臂
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader  # noqa: E402

from data.collate import multimodal_collate_fn  # noqa: E402
from data.dataset import MultimodalDataset  # noqa: E402
from models.multimodal_model import MultimodalEmotionModel  # noqa: E402
from scripts.train import _apply_single_domain_emotion_classes, _batch_to_model_inputs, _subset_by_datasets  # noqa: E402
from utils.helpers import calculate_metrics  # noqa: E402

# 与 EmotionShiftFusion._MODAL_NAMES 的顺序保持一致
MODAL_ORDER = ["video", "audio", "physiological", "text"]


def drop_modality(batch: Dict[str, torch.Tensor], modal: str, keep_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """按 keep_mask（B，1 表示保留）把指定模态置零。"""
    out = dict(batch)
    b = keep_mask.shape[0]
    m = keep_mask.view(b, 1)
    if modal == "video" and out.get("video") is not None:
        v = out["video"]
        out["video"] = v * m.view(b, *([1] * (v.dim() - 1)))
    elif modal == "audio":
        for key in ("audio", "audio_precomputed"):
            if out.get(key) is not None:
                v = out[key]
                out[key] = v * m.view(b, *([1] * (v.dim() - 1)))
    elif modal == "text":
        if out.get("text_input_ids") is not None:
            ids = out["text_input_ids"]
            out["text_input_ids"] = ids * m.view(b, *([1] * (ids.dim() - 1))).long()
        if out.get("text_attention_mask") is not None:
            am = out["text_attention_mask"]
            out["text_attention_mask"] = am * m.view(b, *([1] * (am.dim() - 1))).long()
    return out


def evaluate(model, loader, cfg, device, modal: str | None, rate: float,
             use_mask: bool, seed: int) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    model.eval()
    preds: List[int] = []
    tgts: List[int] = []
    with torch.no_grad():
        for batch in loader:
            b = batch["emotion_label"].shape[0]
            modality_mask = None
            if modal is not None and rate > 0:
                # batch 此时仍在 CPU，掩码必须与之一致；送入模型前再转 device。
                keep = torch.from_numpy((rng.rand(b) >= rate).astype(np.float32))
                batch = drop_modality(batch, modal, keep)
                if use_mask:
                    mm = torch.ones(b, len(MODAL_ORDER))
                    idx = MODAL_ORDER.index(modal)
                    mm[:, idx] = keep.view(b)
                    modality_mask = mm.to(device)
            inputs = _batch_to_model_inputs(batch, device, cfg, use_prev_labels=False)
            if modality_mask is not None:
                inputs["modality_mask"] = modality_mask
            out = model(**inputs)
            logits = out["emotion_logits"]
            preds.extend(logits.argmax(-1).reshape(-1).cpu().tolist())
            tgts.extend(batch["emotion_label"].reshape(-1).cpu().tolist())
    return calculate_metrics(predictions=preds, targets=tgts, task="classification")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="meld")
    ap.add_argument("--rates", default="0,0.15,0.3,0.5")
    ap.add_argument("--modalities", default="text,audio,video")
    ap.add_argument("--use-mask", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default="logs_e4/missing_modality.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg.setdefault("training", {}).setdefault("finetune", {})["datasets"] = [args.dataset]
    _apply_single_domain_emotion_classes(cfg, "finetune")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = MultimodalDataset(cfg["data"]["root_dir"], split="val", config=cfg)
    base = _subset_by_datasets(base, cfg, "finetune")
    loader = DataLoader(base, batch_size=16, shuffle=False, num_workers=0,
                        collate_fn=multimodal_collate_fn)

    model = MultimodalEmotionModel(cfg).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    print(f"[{args.name}] 缺失键={len(missing)} 多余键={len(unexpected)} 验证样本={len(base)}")

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    modals = [m.strip() for m in args.modalities.split(",") if m.strip()]

    rows = []
    base_f1 = None
    m0 = evaluate(model, loader, cfg, device, None, 0.0, args.use_mask, args.seed)
    print(f"[{args.name}] p=0        F1={m0['f1']:.4f} Acc={m0['accuracy']:.4f}")
    rows.append({"arm": args.name, "modality": "none", "rate": 0.0,
                 "f1": round(m0["f1"], 4), "accuracy": round(m0["accuracy"], 4),
                 "delta_f1": 0.0})
    base_f1 = m0["f1"]

    for modal in modals:
        for rate in rates:
            if rate <= 0:
                continue
            r = evaluate(model, loader, cfg, device, modal, rate, args.use_mask, args.seed)
            d = base_f1 - r["f1"]
            print(f"[{args.name}] drop {modal:13s} p={rate:.2f} F1={r['f1']:.4f} Acc={r['accuracy']:.4f} ΔF1={d:+.4f}")
            rows.append({"arm": args.name, "modality": modal, "rate": rate,
                         "f1": round(r["f1"], 4), "accuracy": round(r["accuracy"], 4),
                         "delta_f1": round(d, 4)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["arm", "modality", "rate", "f1", "accuracy", "delta_f1"]
    exists = out.is_file()
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"追加 {len(rows)} 行 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

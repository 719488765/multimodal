#!/usr/bin/env python3
"""MELD checkpoint 评估：Top-1 Acc、macro/weighted 指标、每类 P/R/F1、混淆矩阵。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import MultimodalDataset  # noqa: E402
from data.collate import multimodal_collate_fn  # noqa: E402
from models.multimodal_model import MultimodalEmotionModel  # noqa: E402
from utils.helpers import calculate_metrics, get_dataloader_kwargs, load_config, setup_device  # noqa: E402
from utils.visualization import plot_confusion_matrix  # noqa: E402

EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]


def _subset_meld(dataset, config):
    """与 train.py 一致：仅保留 MELD 样本。"""
    ds_list = (
        config.get("training", {}).get("pretrain", {}).get("datasets")
        or config.get("training", {}).get("finetune", {}).get("datasets")
        or []
    )
    if not ds_list:
        return dataset
    target_ids = {1}  # meld
    if "meld" not in [str(x).lower() for x in ds_list]:
        name_map = {"crema": 0, "meld": 1, "mosei": 2}
        target_ids = {name_map.get(str(x).lower(), -1) for x in ds_list}
    keep = [
        i
        for i, s in enumerate(dataset.data_list)
        if s.get("dataset_id", -1) in target_ids
    ]
    return Subset(dataset, keep) if keep else dataset


def evaluate(config_path: str, checkpoint_path: str, split: str, batch_size: int, out_dir: str):
    config = load_config(config_path)
    device = setup_device(config)

    base_ds = MultimodalDataset(config["data"]["root_dir"], split=split, config=config)
    dataset = _subset_meld(base_ds, config)
    dl_kwargs = get_dataloader_kwargs(config, shuffle=False)
    dl_kwargs["collate_fn"] = multimodal_collate_fn
    loader = DataLoader(dataset, batch_size=batch_size, **dl_kwargs)

    model = MultimodalEmotionModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "video": batch["video"].to(device) if batch.get("video") is not None else None,
                "audio": batch["audio"].to(device) if batch.get("audio") is not None else None,
                "audio_precomputed": batch["audio_precomputed"].to(device)
                if batch.get("audio_precomputed") is not None
                else None,
                "physiological": batch["physiological"].to(device)
                if batch.get("physiological") is not None
                else None,
                "text_input_ids": batch["text_input_ids"].to(device)
                if batch.get("text_input_ids") is not None
                else None,
                "text_attention_mask": batch["text_attention_mask"].to(device)
                if batch.get("text_attention_mask") is not None
                else None,
            }
            outputs = model(**inputs, return_domain_logits=False)
            preds = torch.argmax(outputs["emotion_logits"], dim=1)
            all_targets.extend(batch["emotion_label"].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    metrics = calculate_metrics(predictions=all_preds, targets=all_targets, task="classification")
    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(7)),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(7)))

    per_class = {}
    for i, name in enumerate(EMOTION_NAMES):
        mask_t = np.array(all_targets) == i
        mask_p = np.array(all_preds) == i
        tp = int(np.sum((np.array(all_targets) == i) & (np.array(all_preds) == i)))
        per_class[name] = {
            "support": int(mask_t.sum()),
            "predicted": int(mask_p.sum()),
            "tp": tp,
        }

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    cm_path = os.path.join(out_dir, f"confusion_matrix_{split}_{tag}.png")
    plot_confusion_matrix(all_targets, all_preds, EMOTION_NAMES, cm_path)

    result = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "split": split,
        "n_samples": len(all_targets),
        "metrics_weighted": metrics,
        "per_class_counts": per_class,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": cm_path,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MELD checkpoint with per-class metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="")
    parser.add_argument("--out-dir", default="outputs_accuracy_seq/meld_eval")
    args = parser.parse_args()

    result = evaluate(
        args.config,
        args.checkpoint,
        args.split,
        args.batch_size,
        args.out_dir,
    )

    print(json.dumps(result["metrics_weighted"], ensure_ascii=False, indent=2))
    print("\n=== Per-class classification report ===")
    print(result["classification_report"])
    print(f"Confusion matrix plot: {result['confusion_matrix_plot']}")

    if args.output:
        out = {k: v for k, v in result.items() if k != "classification_report"}
        out["classification_report_text"] = result["classification_report"]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

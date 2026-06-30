#!/usr/bin/env python3
"""泛化 checkpoint 评估：meld / crema / mosei，支持 native / unified 标签空间。"""

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

from data.collate import multimodal_collate_fn
from data.dataset import MultimodalDataset
from models.multimodal_model import MultimodalEmotionModel
from utils.helpers import calculate_metrics, get_dataloader_kwargs, load_config, setup_device
from utils.label_mapping import get_emotion_class_names, uses_native_labels
from utils.visualization import plot_confusion_matrix

DS_NAME_TO_ID = {"crema": 0, "meld": 1, "mosei": 2}


def _primary_dataset(config) -> str | None:
    ds_list = (
        config.get("training", {}).get("pretrain", {}).get("datasets")
        or config.get("training", {}).get("finetune", {}).get("datasets")
        or []
    )
    if len(ds_list) == 1:
        return str(ds_list[0]).lower()
    return None


def _subset_dataset(base_ds, config):
    ds_name = _primary_dataset(config)
    if not ds_name:
        return base_ds
    target_id = DS_NAME_TO_ID.get(ds_name)
    if target_id is None:
        return base_ds
    keep = [
        i
        for i, s in enumerate(base_ds.data_list)
        if s.get("dataset_id", -1) == target_id
    ]
    return Subset(base_ds, keep) if keep else base_ds


def evaluate(
    config_path: str,
    checkpoint_path: str,
    split: str,
    batch_size: int,
    out_dir: str,
):
    config = load_config(config_path)
    device = setup_device(config)
    ds_name = _primary_dataset(config) or "meld"
    num_classes = config.get("model", {}).get("output", {}).get("emotion_classes", 7)
    class_names = get_emotion_class_names(
        ds_name,
        config.get("datasets", {}),
        num_classes=num_classes,
    )
    native = uses_native_labels(ds_name, config.get("datasets", {}))

    base_ds = MultimodalDataset(config["data"]["root_dir"], split=split, config=config)
    dataset = _subset_dataset(base_ds, config)
    dl_kwargs = get_dataloader_kwargs(config, shuffle=False)
    dl_kwargs["collate_fn"] = multimodal_collate_fn
    loader = DataLoader(dataset, batch_size=batch_size, **dl_kwargs)

    model = MultimodalEmotionModel(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    labels = list(range(num_classes))
    metrics = calculate_metrics(
        predictions=all_preds, targets=all_targets, task="classification"
    )
    report = classification_report(
        all_targets,
        all_preds,
        labels=labels,
        target_names=class_names[:num_classes],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets, all_preds, labels=labels)

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    cm_path = os.path.join(out_dir, f"confusion_matrix_{split}_{tag}.png")
    plot_confusion_matrix(all_targets, all_preds, class_names[:num_classes], cm_path)

    return {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "dataset": ds_name,
        "native_labels": native,
        "num_classes": num_classes,
        "class_names": class_names[:num_classes],
        "split": split,
        "n_samples": len(all_targets),
        "metrics_weighted": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": cm_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SDAVT checkpoint (any single-domain dataset)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-dir", default="outputs_sdavt_v3/eval")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = evaluate(
        args.config,
        args.checkpoint,
        args.split,
        args.batch_size,
        args.out_dir,
    )

    print(json.dumps(result["metrics_weighted"], ensure_ascii=False, indent=2))
    print(f"\nDataset: {result['dataset']} native={result['native_labels']} classes={result['num_classes']}")
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

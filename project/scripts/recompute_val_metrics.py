"""
Author: AI
Date: 2026-03-10
Description: 从指定 checkpoint 重算验证集指标（accuracy/precision/recall/f1）
"""

import argparse
import os
import sys
import json

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径，便于导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_config, setup_device, calculate_metrics  # noqa: E402
from data.dataset import MultimodalDataset  # noqa: E402
from data.collate import multimodal_collate_fn  # noqa: E402
from models.multimodal_model import MultimodalEmotionModel  # noqa: E402


def recompute_metrics(config_path: str, checkpoint_path: str, split: str = "val", batch_size: int = 1):
    config = load_config(config_path)
    device = setup_device(config)

    dataset = MultimodalDataset(config["data"]["root_dir"], split=split, config=config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=multimodal_collate_fn,
    )

    model = MultimodalEmotionModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", {}), strict=False)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "video": batch["video"].to(device) if batch.get("video") is not None else None,
                "audio": batch["audio"].to(device) if batch.get("audio") is not None else None,
                "audio_precomputed": batch["audio_precomputed"].to(device)
                if batch.get("audio_precomputed") is not None
                else None,
                "physiological": batch["physiological"].to(device) if batch.get("physiological") is not None else None,
                "text_input_ids": batch["text_input_ids"].to(device) if batch.get("text_input_ids") is not None else None,
                "text_attention_mask": batch["text_attention_mask"].to(device) if batch.get("text_attention_mask") is not None else None,
            }

            outputs = model(**inputs, return_domain_logits=False)
            preds = torch.argmax(outputs["emotion_logits"], dim=1)
            targets = batch["emotion_label"]
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    metrics = calculate_metrics(predictions=all_preds, targets=all_targets, task="classification")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="从 checkpoint 重算验证指标")
    parser.add_argument("--config", required=True, help="配置文件路径，如 config/config_AVT_noDA.yaml")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径，如 checkpoints/checkpoint_pretrain_epoch_25.pth")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="重算的数据集划分，默认 val")
    parser.add_argument("--batch_size", type=int, default=1, help="评估 batch size，默认 1")
    parser.add_argument("--output", default="", help="可选：输出 JSON 文件路径")
    args = parser.parse_args()

    metrics = recompute_metrics(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

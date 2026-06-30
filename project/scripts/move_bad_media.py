#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: AI-Assistant
# Date: 2026-03-10
# Description: 将基于目录结构检测出的坏样本音视频文件移动到 data/bad/ 目录

"""
坏样本移动脚本（配合 bad_samples_*_dir.csv 使用）

用途：
- 在已经通过 `scripts/check_media_health_dir.py` 生成：
    data/bad_samples_train_dir.csv
    data/bad_samples_val_dir.csv
  之后，读取这些 CSV，将其中列出的 video_path / audio_path
  物理移动到 data/bad/{split}/video/ 与 data/bad/{split}/audio/ 目录。

效果：
- 原始 data/{split}/video 与 data/{split}/audio 中不再包含坏样本文件，
  下次训练遍历目录时会自然跳过这些样本。

安全说明：
- 仅做“移动”，不做删除；如需恢复，可从 data/bad/... 再移回原目录。
"""

import os
import csv
import argparse
import logging
from typing import List, Dict

import shutil


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_bad_samples(csv_path: str) -> List[Dict[str, str]]:
    """读取 bad_samples_{split}_dir.csv，返回记录列表。"""
    records: List[Dict[str, str]] = []
    if not os.path.exists(csv_path):
        logger.warning("未找到坏样本列表: %s，跳过。", csv_path)
        return records

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    logger.info("加载坏样本列表: %s (样本数=%d)", csv_path, len(records))
    return records


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_file_safe(src: str, dst_dir: str) -> None:
    """安全移动文件到目标目录（若不存在则跳过并记录日志）。"""
    if not src:
        return
    if not os.path.exists(src):
        logger.warning("文件不存在，无法移动: %s", src)
        return

    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src))
    logger.info("移动文件: %s -> %s", src, dst_path)
    shutil.move(src, dst_path)


def process_split(data_dir: str, split: str) -> None:
    """
    针对单个划分（train / val），根据 bad_samples_{split}_dir.csv
    将坏样本的音视频文件移动到 data/bad/{split}/video 或 audio 下。
    """
    bad_csv = os.path.join(data_dir, f"bad_samples_{split}_dir.csv")
    records = load_bad_samples(bad_csv)
    if not records:
        logger.info("未检测到 %s 划分的坏样本或文件不存在，跳过移动。", split)
        return

    bad_root = os.path.join(data_dir, "bad", split)
    bad_video_dir = os.path.join(bad_root, "video")
    bad_audio_dir = os.path.join(bad_root, "audio")

    for rec in records:
        video_path = rec.get("video_path", "")
        audio_path = rec.get("audio_path", "")

        if video_path:
            move_file_safe(video_path, bad_video_dir)
        if audio_path:
            move_file_safe(audio_path, bad_audio_dir)


def main() -> None:
    """
    命令行入口。

    示例用法：
        python scripts/move_bad_media.py --data_dir data
    """
    parser = argparse.ArgumentParser(description="将坏样本音视频文件移动到 data/bad/... 目录")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据根目录（包含 train/val 以及 bad_samples_*_dir.csv），默认 ./data",
    )

    args = parser.parse_args()
    setup_logging()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        logger.error("数据目录不存在: %s", data_dir)
        return

    logger.info("开始移动坏样本文件, data_dir=%s", data_dir)
    for split in ["train", "val"]:
        process_split(data_dir=data_dir, split=split)

    logger.info("移动完成。坏样本已归档至 data/bad/{train,val}/{video,audio}/。")


if __name__ == "__main__":
    main()


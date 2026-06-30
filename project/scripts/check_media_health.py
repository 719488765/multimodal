#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: AI-Assistant
# Date: 2026-03-10
# Description: 多模态音视频数据健康检查与坏样本列表导出脚本

"""
多模态音视频数据健康检查脚本

用途（只读、安全）：
- 遍历 data/ 目录下的 train / val 划分
- 对每条样本的 video_path / audio_path 做实际解码尝试
- 识别出无法正常打开或无法解码出帧/波形的“坏样本”
- 为每个 split 生成：
  - bad_samples_{split}.csv : 仅包含坏样本
  - clean_{split}.csv       : 过滤掉坏样本后的干净列表

说明：
- 不会修改或删除任何原始文件，仅生成新的 CSV
- 训练时如果希望跳过坏样本，可以让配置指向 clean_train.csv / clean_val.csv
"""

import os
import csv
import argparse
import logging
from typing import Dict, List, Tuple

import cv2  # type: ignore
import librosa  # type: ignore


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    初始化简单日志配置，既输出到控制台也输出到文件（可选）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def check_video_health(video_path: str) -> Tuple[bool, str]:
    """
    检查单个视频文件是否健康可读。

    返回:
        (is_ok, reason)
        - is_ok: True 表示可正常解码至少一帧；False 表示坏样本
        - reason: 当 is_ok=False 时，给出失败原因描述
    """
    if not video_path:
        return True, ""  # 无视频路径视为当前任务不关心该模态

    if not os.path.exists(video_path):
        return False, "video_path_not_exists"

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # 典型: ffmpeg 底层 moov atom not found 等问题
            return False, "video_cannot_open"

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return False, "video_no_frame_decoded"

        return True, ""
    except Exception as e:
        return False, f"video_exception:{repr(e)}"


def check_audio_health(audio_path: str, sample_rate: int = 16000) -> Tuple[bool, str]:
    """
    检查单个音频文件是否健康可读。

    返回:
        (is_ok, reason)
        - is_ok: True 表示 librosa 能正常加载；False 表示坏样本
        - reason: 当 is_ok=False 时，给出失败原因描述
    """
    if not audio_path:
        return True, ""  # 无音频路径视为当前任务不关心该模态

    if not os.path.exists(audio_path):
        return False, "audio_path_not_exists"

    try:
        # 读一小段即可验证封装是否正常
        librosa.load(audio_path, sr=sample_rate, duration=1.0)
        return True, ""
    except Exception as e:
        return False, f"audio_exception:{repr(e)}"


def load_csv_records(csv_path: str) -> List[Dict[str, str]]:
    """
    读取原始 CSV（train.csv / val.csv），返回字典列表。
    """
    records: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def save_csv_records(csv_path: str, records: List[Dict[str, str]]) -> None:
    """
    以原表头字段顺序写回 CSV。
    """
    if not records:
        logger.warning("目标记录为空，跳过写入: %s", csv_path)
        return

    fieldnames = list(records[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def process_split(
    data_dir: str,
    split: str,
    sample_rate: int = 16000,
) -> None:
    """
    针对单个划分（train / val），扫描并输出坏样本 / 干净样本列表。
    """
    csv_path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(csv_path):
        logger.warning("未找到 %s，对应 %s 划分将被跳过。", csv_path, split)
        return

    logger.info("开始检查 %s 划分: %s", split, csv_path)
    records = load_csv_records(csv_path)

    bad_records: List[Dict[str, str]] = []
    clean_records: List[Dict[str, str]] = []

    total = len(records)
    for idx, rec in enumerate(records, start=1):
        sample_id = rec.get("sample_id", f"{split}_{idx}")
        video_path = rec.get("video_path", "")
        audio_path = rec.get("audio_path", "")

        # 默认认为样本健康
        is_ok = True
        reasons: List[str] = []

        v_ok, v_reason = check_video_health(video_path)
        if not v_ok:
            is_ok = False
            reasons.append(v_reason)

        a_ok, a_reason = check_audio_health(audio_path, sample_rate=sample_rate)
        if not a_ok:
            is_ok = False
            reasons.append(a_reason)

        if not is_ok:
            rec_with_reason = dict(rec)
            rec_with_reason["bad_reason"] = ";".join(reasons)
            bad_records.append(rec_with_reason)
            logger.debug(
                "检测到坏样本: split=%s, sample_id=%s, reason=%s",
                split,
                sample_id,
                rec_with_reason["bad_reason"],
            )
        else:
            clean_records.append(rec)

        if idx % 100 == 0 or idx == total:
            logger.info(
                "进度 %s: %d / %d，当前坏样本数=%d",
                split,
                idx,
                total,
                len(bad_records),
            )

    # 导出结果 CSV（与原始 CSV 同目录，文件名带前缀以示区分）
    bad_csv = os.path.join(data_dir, f"bad_samples_{split}.csv")
    clean_csv = os.path.join(data_dir, f"clean_{split}.csv")

    if bad_records:
        save_csv_records(bad_csv, bad_records)
        logger.info("坏样本列表已保存: %s (数量=%d)", bad_csv, len(bad_records))
    else:
        logger.info("未检测到坏样本: %s", split)

    if clean_records:
        save_csv_records(clean_csv, clean_records)
        logger.info("干净样本列表已保存: %s (数量=%d)", clean_csv, len(clean_records))
    else:
        logger.warning("所有样本都被判定为坏样本，请检查数据与检测逻辑: %s", split)


def main() -> None:
    """
    命令行入口。

    示例用法：
        python scripts/check_media_health.py --data_dir data

    扫描 data/train.csv 和 data/val.csv，生成:
        data/bad_samples_train.csv
        data/clean_train.csv
        data/bad_samples_val.csv
        data/clean_val.csv
    """
    parser = argparse.ArgumentParser(description="多模态音视频数据健康检查脚本（只读）")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据根目录（包含 train.csv / val.csv），默认 ./data",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="音频采样率（用于 librosa.load），默认 16000",
    )

    args = parser.parse_args()

    setup_logging()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        logger.error("数据目录不存在: %s", data_dir)
        return

    logger.info("开始多模态音视频健康检查, data_dir=%s", data_dir)

    for split in ["train", "val"]:
        process_split(data_dir=data_dir, split=split, sample_rate=args.sample_rate)

    logger.info("检查完成，你可以在 %s 目录下查看 clean_*.csv 与 bad_samples_*.csv。", data_dir)


if __name__ == "__main__":
    main()


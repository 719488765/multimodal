#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: AI-Assistant
# Date: 2026-03-10
# Description: 基于目录结构的多模态音视频健康检查脚本（只读版）

"""
基于目录结构的多模态音视频健康检查脚本（只读版）

适用场景：
- 使用 `project/data/dataset.py` 中的“方式二：目录结构”组织数据，即：
  data/{train,val,test}/{video,audio,physiological,text,labels}
- 不依赖 train.csv / val.csv，而是通过文件名 sample_id 推断各模态路径。

功能：
- 针对 train / val 划分：
  - 扫描 data/{split}/video/ 下的所有视频文件，提取 sample_id（去掉扩展名）
  - 为每个 sample_id 组装对应的 audio 路径（如存在）
  - 使用 OpenCV 与 Librosa 尝试实际解码
  - 将无法正常打开/解码的视频或音频样本记录到：
      data/bad_samples_{split}_dir.csv

说明：
- 本脚本为只读版，不会修改或删除任何文件。
- 后续可配合单独的移动脚本，将坏样本移动到 data/bad/...，从而在训练时自动跳过。
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
    """初始化日志配置。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def detect_supported_video(filename: str) -> bool:
    """判断文件名是否是支持的视频格式。"""
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv"))


def detect_supported_audio(filename: str) -> bool:
    """判断文件名是否是支持的音频格式。"""
    return filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))


def check_video_health(video_path: str) -> Tuple[bool, str]:
    """
    检查单个视频文件是否健康可读。
    返回:
        (is_ok, reason)
    """
    if not os.path.exists(video_path):
        return False, "video_path_not_exists"

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
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
    """
    if not os.path.exists(audio_path):
        return False, "audio_path_not_exists"

    try:
        # 读一小段即可验证封装是否正常
        librosa.load(audio_path, sr=sample_rate, duration=1.0)
        return True, ""
    except Exception as e:
        return False, f"audio_exception:{repr(e)}"


def collect_samples_from_dir(data_dir: str, split: str) -> List[Dict[str, str]]:
    """
    从 data/{split}/video 与 data/{split}/audio 目录中收集样本列表。
    返回字典列表，每个元素至少包含：
        - split
        - sample_id
        - video_path（如存在）
        - audio_path（如存在）
    """
    split_dir = os.path.join(data_dir, split)
    video_dir = os.path.join(split_dir, "video")
    audio_dir = os.path.join(split_dir, "audio")

    samples: Dict[str, Dict[str, str]] = {}

    # 收集视频样本
    if os.path.exists(video_dir):
        for filename in os.listdir(video_dir):
            if not detect_supported_video(filename):
                continue
            sample_id = os.path.splitext(filename)[0]
            video_path = os.path.join(video_dir, filename)
            rec = samples.setdefault(
                sample_id,
                {"split": split, "sample_id": sample_id, "video_path": "", "audio_path": ""},
            )
            rec["video_path"] = video_path

    # 收集音频样本（与 sample_id 对齐）
    if os.path.exists(audio_dir):
        for filename in os.listdir(audio_dir):
            if not detect_supported_audio(filename):
                continue
            sample_id = os.path.splitext(filename)[0]
            audio_path = os.path.join(audio_dir, filename)
            rec = samples.setdefault(
                sample_id,
                {"split": split, "sample_id": sample_id, "video_path": "", "audio_path": ""},
            )
            rec["audio_path"] = audio_path

    return list(samples.values())


def save_bad_samples_csv(csv_path: str, records: List[Dict[str, str]]) -> None:
    """将坏样本记录写入 CSV。"""
    if not records:
        logger.info("未检测到坏样本，无需写入: %s", csv_path)
        return

    fieldnames = ["split", "sample_id", "video_path", "audio_path", "bad_reason"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def process_split_dir(data_dir: str, split: str, sample_rate: int = 16000) -> None:
    """针对目录结构的单个划分（train / val）执行健康检查。"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        logger.warning("未找到目录: %s，对应 %s 划分将被跳过。", split_dir, split)
        return

    logger.info("开始基于目录结构检查 %s 划分, 根目录=%s", split, split_dir)
    samples = collect_samples_from_dir(data_dir, split)
    total = len(samples)
    logger.info("%s 划分共发现样本数: %d（以 video/audio 文件名为基准的 sample_id 去重后）", split, total)

    bad_records: List[Dict[str, str]] = []

    for idx, rec in enumerate(samples, start=1):
        sample_id = rec["sample_id"]
        video_path = rec.get("video_path", "")
        audio_path = rec.get("audio_path", "")

        is_ok = True
        reasons: List[str] = []

        # 如果存在视频文件，则检查视频健康性
        if video_path:
            v_ok, v_reason = check_video_health(video_path)
            if not v_ok:
                is_ok = False
                reasons.append(v_reason)

        # 如果存在音频文件，则检查音频健康性
        if audio_path:
            a_ok, a_reason = check_audio_health(audio_path, sample_rate=sample_rate)
            if not a_ok:
                is_ok = False
                reasons.append(a_reason)

        if not is_ok:
            bad_rec = {
                "split": split,
                "sample_id": sample_id,
                "video_path": video_path,
                "audio_path": audio_path,
                "bad_reason": ";".join(reasons),
            }
            bad_records.append(bad_rec)

        if idx % 200 == 0 or idx == total:
            logger.info(
                "进度 %s: %d / %d，当前坏样本数=%d",
                split,
                idx,
                total,
                len(bad_records),
            )

    # 写出坏样本 CSV
    bad_csv = os.path.join(data_dir, f"bad_samples_{split}_dir.csv")
    save_bad_samples_csv(bad_csv, bad_records)
    logger.info("目录结构坏样本列表已保存: %s (数量=%d)", bad_csv, len(bad_records))


def main() -> None:
    """
    命令行入口。

    示例用法：
        python scripts/check_media_health_dir.py --data_dir data

    将在 data/ 目录下生成：
        bad_samples_train_dir.csv
        bad_samples_val_dir.csv
    """
    parser = argparse.ArgumentParser(description="基于目录结构的多模态音视频健康检查脚本（只读版）")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据根目录（包含 train/val/test 子目录），默认 ./data",
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

    logger.info("开始基于目录结构的多模态音视频健康检查, data_dir=%s", data_dir)
    for split in ["train", "val"]:
        process_split_dir(data_dir=data_dir, split=split, sample_rate=args.sample_rate)

    logger.info("检查完成，你可以在 %s 目录下查看 bad_samples_*_dir.csv。", data_dir)


if __name__ == "__main__":
    main()


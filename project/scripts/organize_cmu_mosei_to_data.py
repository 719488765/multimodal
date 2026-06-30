#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMU-MOSEI 数据集整合脚本（基于 SDK 特征）

目的：
- 将通过 CMU-MultimodalSDK 下载到本地的 CMU-MOSEI 标签信息，
  整合为本项目现有的 data 目录结构，方便直接使用现有的 MultimodalDataset 和 train.py。

重要说明：
- 官方 SDK 和我们当前的下载，仅提供特征与标签（.csd），不包含原始 YouTube 视频文件；
- 因版权原因，我们无法通过 SDK 恢复出真实的视频 / 原始语音；
- 本脚本会为每个样本创建：
    - 文本文件：占位文本（后续如有转录文本可替换）
    - 标签文件：目前统一写为 neutral + (0.0, 0.0)，占位用；
  视频 / 音频 / 生理信号暂不生成文件，模型会自动使用零特征占位。

这意味着：集成后的 CMU-MOSEI 样本目前只能为训练提供“数据集 ID + 占位文本 + 占位标签”，
适合先打通训练流程，未来如你获取到真实视频 / 文本 / 标签，可在本脚本基础上完善。
"""

import os
from pathlib import Path

from mmsdk import mmdatasdk


# 项目根目录
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"

# 已通过 organize_cmu_mosei.py 整理好的 SDK 数据目录
MOSEI_SDK_ROOT = os.path.join(PROJECT_ROOT, "data", "CMU_MOSEI")
HIGHLEVEL_DIR = os.path.join(MOSEI_SDK_ROOT, "highlevel")
LABELS_DIR = os.path.join(MOSEI_SDK_ROOT, "labels")

# 目标：按照 MultimodalDataset 期望的结构整合到该目录
TARGET_ROOT = os.path.join(PROJECT_ROOT, "data", "CMU_MOSEI_MM")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_mosei_labels_dataset():
    """
    仅基于 labels 构建 mmdataset，用于遍历每个视频及其分段。
    """
    if not os.path.isdir(LABELS_DIR):
        raise RuntimeError(f"标签目录不存在，请先运行 organize_cmu_mosei.py：{LABELS_DIR}")

    print(f"从 {LABELS_DIR} 加载 CMU-MOSEI 标签 .csd 文件...")
    # mmdataset 的第二个参数是数据根目录；这里仅使用 labels 字段
    ds = mmdatasdk.mmdataset(
        mmdatasdk.cmu_mosei.labels,
        LABELS_DIR,
    )

    print("可用的 computational_sequences 键：", list(ds.computational_sequences.keys()))
    return ds


def get_standard_folds():
    """
    使用 SDK 提供的标准划分，将样本划分为 train / val / test。
    """
    # 不同版本中字段名可能有差异，如果报错，请在交互式 Python 中打印 dir(mmdatasdk.cmu_mosei)
    folds = mmdatasdk.cmu_mosei.standard_folds  # type: ignore

    split_dict = {
        "train": set(folds["train"]),
        "val": set(folds.get("valid", folds.get("dev", []))),
        "test": set(folds["test"]),
    }
    return split_dict


def organize_cmu_mosei_to_data():
    """
    将 CMU-MOSEI 标签整合为本项目 data 目录下的标准结构：

    data/CMU_MOSEI_MM/
      ├── train/
      │   ├── video/
      │   ├── audio/
      │   ├── physiological/
      │   ├── text/
      │   └── labels/
      ├── val/   # 同上
      └── test/  # 同上
    """
    print("=" * 60)
    print("CMU-MOSEI -> data 目录整合脚本（基于 SDK labels）")
    print("=" * 60)

    # 1. 加载标签 mmdataset + 标准划分
    labels_ds = load_mosei_labels_dataset()
    folds = get_standard_folds()

    # 2. 预创建目标目录结构
    for split in ["train", "val", "test"]:
        for sub in ["video", "audio", "physiological", "text", "labels"]:
            ensure_dir(os.path.join(TARGET_ROOT, split, sub))

    # 3. 选择 labels 的 csd 名称（通常只有一个 key）
    if not labels_ds.computational_sequences:
        raise RuntimeError("labels mmdataset 中没有任何 computational_sequences，无法继续。")

    labels_key = list(labels_ds.computational_sequences.keys())[0]
    labels_csd = labels_ds.computational_sequences[labels_key]

    # labels_csd['data'] 是一个 dict: {video_id: {'features': (num_segments, dim), 'intervals': (num_segments, 2)}}
    all_video_ids = list(labels_csd["data"].keys())
    print(f"检测到视频条目数: {len(all_video_ids)}")

    counters = {"train": 0, "val": 0, "test": 0}

    for vid in all_video_ids:
        # 根据标准划分确定 video 属于哪个 split
        split = None
        for s in ["train", "val", "test"]:
            if vid in folds[s]:
                split = s
                break
        if split is None:
            # 未出现在标准划分中的样本，跳过
            continue

        vid_data = labels_csd["data"][vid]
        feats = vid_data["features"]
        num_segments = feats.shape[0]

        for seg_idx in range(num_segments):
            counters[split] += 1
            sample_id = f"mosei_{split}_{counters[split]:06d}"

            # 3.1 文本文件：当前仅写入占位文本
            text_path = os.path.join(TARGET_ROOT, split, "text", f"{sample_id}.txt")
            if not os.path.exists(text_path):
                with open(text_path, "w", encoding="utf-8") as f_txt:
                    f_txt.write(f"Placeholder transcript for {vid} segment {seg_idx}\n")

            # 3.2 标签文件：目前写入占位标签（neutral + 0.0,0.0）
            label_path = os.path.join(TARGET_ROOT, split, "labels", f"{sample_id}.txt")
            if not os.path.exists(label_path):
                with open(label_path, "w", encoding="utf-8") as f_lab:
                    f_lab.write("neutral\n")
                    f_lab.write("0.0,0.0\n")

            # 视频 / 音频 / 生理信号：暂不创建文件，MultimodalDataset 会自动得到 None，
            # 模型内部会用零向量占位。

    print("\n整合完成！各划分样本数：")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counters[split]} 个样本")

    print(f"\n数据已整理到：{TARGET_ROOT}")
    print("你可以在配置文件 config.yaml 中，将 data.root_dir 设置为该路径进行训练。")


def main():
    try:
        organize_cmu_mosei_to_data()
    except Exception as e:
        print("\n错误：整理 CMU-MOSEI 过程中出现异常：")
        print(e)
        print("\n请确认：")
        print(f"  1) 已经运行过 organize_cmu_mosei.py，使 SDK 数据位于 {MOSEI_SDK_ROOT}")
        print(f"  2) {LABELS_DIR} 下存在 CMU_MOSEI_Labels.csd 等标签文件")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMU-MOSEI 数据集整理脚本

功能：
- 使用已经安装好的 CMU-MultimodalSDK (mmsdk) 下载 / 检查 CMU-MOSEI 的 highlevel 特征和标签；
- 将下载得到的特征和标签“整合”到统一的数据目录：
    /home/lizhichun_24/sda1/code/multimodal/project/data/CMU_MOSEI

重要说明：
- CMU-MOSEI SDK 只提供处理好的特征（文本 / 音频 / 视觉等）和标签，不包含原始 YouTube 视频文件；
- 本脚本不会生成项目中 MultimodalDataset 所用的那种 video/audio/text 目录结构，
  而是将 SDK 的 highlevel / labels 统一集中到 project/data/CMU_MOSEI，方便后续单独使用。
"""

import os
import shutil
from pathlib import Path

from mmsdk import mmdatasdk


# 项目根目录与路径配置
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"

# SDK 下载保存路径（原始 SDK 脚本使用的目录）
DOWNLOAD_ROOT = os.path.join(PROJECT_ROOT, "downloads", "CMU_MOSEI_SDK")

# 统一数据根目录：所有整理好的数据集都放在这里
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# 本脚本为 MOSEI 预留的统一目录
MOSEI_DATA_DIR = os.path.join(DATA_ROOT, "CMU_MOSEI")


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def download_cmu_mosei_via_sdk():
    """
    使用 mmsdk 下载 / 补全 CMU-MOSEI 的 highlevel 特征和 labels。
    - 如果 highlevel 目录已经存在且非空，只补全 labels；
    - 如果 labels 目录已经存在且非空，只补全 highlevel；
    - 如果两者都存在且非空，则全部跳过。
    """
    print("=" * 60)
    print("检查 / 下载 CMU-MOSEI SDK 数据")
    print("=" * 60)

    ensure_dir(DOWNLOAD_ROOT)

    highlevel_dir = os.path.join(DOWNLOAD_ROOT, "highlevel")
    labels_dir = os.path.join(DOWNLOAD_ROOT, "labels")

    highlevel_exists = os.path.isdir(highlevel_dir) and len(os.listdir(highlevel_dir)) > 0
    labels_exists = os.path.isdir(labels_dir) and len(os.listdir(labels_dir)) > 0

    if highlevel_exists and labels_exists:
        print(f"检测到已存在的 highlevel 与 labels 目录：")
        print(f"  highlevel: {highlevel_dir}")
        print(f"  labels  : {labels_dir}")
        print("跳过下载步骤。")
        return highlevel_dir, labels_dir

    print(f"数据将下载到: {DOWNLOAD_ROOT}")

    # 如有需要，下载 highlevel 特征
    if not highlevel_exists:
        print("\n开始创建 CMU-MOSEI 高层特征数据集对象 (highlevel)...")
        ensure_dir(highlevel_dir)
        # 在当前 mmsdk 版本中，mmdataset 初始化时就会自动拉取 / 读取 .csd 文件，
        # 无需再显式调用 download()
        _ = mmdatasdk.mmdataset(
            mmdatasdk.cmu_mosei.highlevel,
            highlevel_dir,
        )
        print("highlevel 特征已就绪。")
    else:
        print("\nhighlevel 目录已存在且非空，跳过 highlevel 下载。")

    # 如有需要，单独下载 labels
    if not labels_exists:
        print("\n开始创建 CMU-MOSEI 标签数据集对象 (labels)...")
        ensure_dir(labels_dir)
        # 同样地，初始化 mmdataset 即可完成 .csd 文件的下载 / 读取
        _ = mmdatasdk.mmdataset(
            mmdatasdk.cmu_mosei.labels,
            labels_dir,
        )
        print("labels 已就绪。")
    else:
        print("\nlabels 目录已存在且非空，跳过 labels 下载。")

    print("\n所有需要的部分已下载完成。")
    print(f"highlevel 目录: {highlevel_dir}")
    print(f"labels   目录: {labels_dir}")

    return highlevel_dir, labels_dir


def safe_symlink(src: str, dst: str):
    """
    创建软链接：dst -> src
    - 如果 dst 已经是正确的链接，什么都不做；
    - 如果 dst 已经存在且不是链接，提示用户手动处理，再决定是否删除/覆盖。
    """
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        print(f"警告：源路径不存在，无法创建软链接：{src}")
        return

    if dst_path.is_symlink():
        # 已经是软链接
        current_target = os.readlink(dst_path)
        if current_target == str(src_path):
            print(f"软链接已存在且指向正确：{dst} -> {src}")
            return
        else:
            print(f"警告：目标已是软链接，但指向 {current_target}，而非 {src}")
            print("如需修正，请先手动删除该链接：")
            print(f"  rm {dst}")
            return

    if dst_path.exists():
        print(f"警告：目标路径已存在且不是软链接：{dst}")
        print("为避免误删，本脚本不会自动覆盖该目录。")
        print("如确认无用，可以手动删除后再次运行本脚本：")
        print(f"  rm -rf {dst}")
        return

    # 创建父目录
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"创建软链接：{dst} -> {src}")
    os.symlink(src_path, dst_path)


def organize_cmu_mosei(use_copy: bool = False):
    """
    将下载好的 CMU-MOSEI SDK 数据整理/集成到统一的数据目录：
        project/data/CMU_MOSEI

    参数：
        use_copy: 为 True 时，使用复制 (cp) 而不是软链接；
                  为 False 时（默认），使用软链接节省磁盘空间。
    """
    print("=" * 60)
    print("整理 / 整合 CMU-MOSEI 数据集到统一 data 目录")
    print("=" * 60)

    highlevel_dir, labels_dir = download_cmu_mosei_via_sdk()

    # 确保统一数据目录存在
    ensure_dir(MOSEI_DATA_DIR)

    target_highlevel = os.path.join(MOSEI_DATA_DIR, "highlevel")
    target_labels = os.path.join(MOSEI_DATA_DIR, "labels")

    if use_copy:
        # 方式一：复制文件（占空间多，但不依赖软链接）
        print("\n使用复制模式（会占用额外磁盘空间）...")

        if not os.path.exists(target_highlevel):
            print(f"复制 highlevel 到: {target_highlevel}")
            shutil.copytree(highlevel_dir, target_highlevel)
        else:
            print(f"目标 highlevel 已存在，跳过复制: {target_highlevel}")

        if not os.path.exists(target_labels):
            print(f"复制 labels 到: {target_labels}")
            shutil.copytree(labels_dir, target_labels)
        else:
            print(f"目标 labels 已存在，跳过复制: {target_labels}")
    else:
        # 方式二：创建软链接（推荐）
        print("\n使用软链接模式（推荐，占用空间小）...")
        safe_symlink(highlevel_dir, target_highlevel)
        safe_symlink(labels_dir, target_labels)

    print("\n整理完成！当前 CMU_MOSEI 统一目录结构：")
    print(f"  {MOSEI_DATA_DIR}/")
    print(f"    ├── highlevel/   -> {highlevel_dir}")
    print(f"    └── labels/      -> {labels_dir}")

    print("\n后续使用示例（Python 代码片段）：")
    print("-" * 60)
    print("from mmsdk import mmdatasdk")
    print(f"data_root = r\"{MOSEI_DATA_DIR}/highlevel\"")
    print("cmu_mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, data_root)")
    print("-" * 60)
    print("注意：如果你想基于这些特征自定义 Dataset，可以在此目录基础上进行开发。")


def main():
    """
    命令行入口：

    示例用法：
        1）默认（推荐软链接）：
            python organize_cmu_mosei.py

        2）使用复制模式（不想用软链接时）：
            python organize_cmu_mosei.py --copy
    """
    import sys

    use_copy = "--copy" in sys.argv

    if use_copy:
        print("使用复制模式 (--copy)，会占用额外磁盘空间。")
    else:
        print("使用软链接模式（默认），不会复制大文件，只创建指向 downloads 的链接。")

    organize_cmu_mosei(use_copy=use_copy)


if __name__ == "__main__":
    main()


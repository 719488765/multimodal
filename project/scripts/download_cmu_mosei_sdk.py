#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 CMU-MultimodalSDK 在服务器上下载 CMU-MOSEI 数据集的对齐特征与标签。

说明：
- SDK 不提供原始 YouTube 视频，只提供已处理好的特征（文本 / 音频 / 视觉）和标签。
- 这些特征适合用来做表示学习或预训练，体积远小于完整 32G 原始视频。
"""

import os
from mmsdk import mmdatasdk


# 项目根目录与下载路径
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"
DOWNLOAD_ROOT = os.path.join(PROJECT_ROOT, "downloads", "CMU_MOSEI_SDK")
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)


def main():
    print("=" * 60)
    print("CMU-MOSEI SDK 下载脚本")
    print("=" * 60)
    print("数据将保存到:", DOWNLOAD_ROOT)

    # 打印可用字段，方便核对 highlevel / labels 等名称
    print("\n可用的 cmu_mosei 字段：")
    print(dir(mmdatasdk.cmu_mosei))

    # 典型用法：下载 highlevel 特征 + labels
    # 注意：如果下面字段名在你环境中不存在，请根据上面的 dir 输出进行调整
    print("\n开始创建 CMU-MOSEI 高层特征数据集对象...")
    highlevel_dir = os.path.join(DOWNLOAD_ROOT, "highlevel")
    cmumosei_highlevel = mmdatasdk.mmdataset(
        mmdatasdk.cmu_mosei.highlevel,  # 若报 AttributeError，请根据 dir 结果修改字段名
        highlevel_dir,
    )

    print("开始追加标签（sentiment / emotion 等）...")
    labels_dir = os.path.join(DOWNLOAD_ROOT, "labels")
    cmumosei_highlevel.add_computational_sequences(
        mmdatasdk.cmu_mosei.labels,  # 同样根据实际字段名调整
        labels_dir,
    )

    print("\n开始下载数据（可能需要较长时间，取决于网络和磁盘空间）...")
    cmumosei_highlevel.download()

    print("\n下载完成！特征与标签已保存到:", DOWNLOAD_ROOT)
    print("highlevel 目录:", highlevel_dir)
    print("labels   目录:", labels_dir)


if __name__ == "__main__":
    main()



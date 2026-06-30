#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 CMU-MOSEI 数据整合到统一的 data 目录

目的：
- 将 data/CMU_MOSEI_MM/ 下的数据合并到 data/train/val/test/ 目录
- 与其他数据集（CREMA-D, MELD）统一管理
- 确保文件名不冲突（使用 mosei_ 前缀）
"""

import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"

# 源目录（CMU-MOSEI 数据）
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "CMU_MOSEI_MM")

# 目标目录（统一的数据目录）
TARGET_ROOT = os.path.join(PROJECT_ROOT, "data")


def merge_dataset():
    """
    将 CMU-MOSEI 数据整合到统一的 data 目录
    """
    print("=" * 60)
    print("CMU-MOSEI 数据整合脚本")
    print("将 data/CMU_MOSEI_MM/ 合并到 data/train/val/test/")
    print("=" * 60)
    
    if not os.path.exists(SOURCE_ROOT):
        print(f"错误：源目录不存在: {SOURCE_ROOT}")
        return
    
    # 检查目标目录是否存在
    for split in ['train', 'val', 'test']:
        target_split_dir = os.path.join(TARGET_ROOT, split)
        if not os.path.exists(target_split_dir):
            print(f"警告：目标目录不存在，创建: {target_split_dir}")
            os.makedirs(target_split_dir, exist_ok=True)
    
    stats = {'train': {'video': 0, 'audio': 0, 'text': 0, 'labels': 0, 'physiological': 0},
             'val': {'video': 0, 'audio': 0, 'text': 0, 'labels': 0, 'physiological': 0},
             'test': {'video': 0, 'audio': 0, 'text': 0, 'labels': 0, 'physiological': 0}}
    
    # 处理每个 split
    for split in ['train', 'val', 'test']:
        print(f"\n处理 {split} 集...")
        
        source_split_dir = os.path.join(SOURCE_ROOT, split)
        target_split_dir = os.path.join(TARGET_ROOT, split)
        
        if not os.path.exists(source_split_dir):
            print(f"  跳过：源目录不存在: {source_split_dir}")
            continue
        
        # 处理每个子目录
        for subdir in ['video', 'audio', 'text', 'labels', 'physiological']:
            source_subdir = os.path.join(source_split_dir, subdir)
            target_subdir = os.path.join(target_split_dir, subdir)
            
            if not os.path.exists(source_subdir):
                continue
            
            # 确保目标子目录存在
            os.makedirs(target_subdir, exist_ok=True)
            
            # 复制文件
            files = os.listdir(source_subdir)
            for filename in files:
                if not filename.startswith('mosei_'):
                    continue  # 只处理 mosei_ 开头的文件
                
                source_file = os.path.join(source_subdir, filename)
                target_file = os.path.join(target_subdir, filename)
                
                # 如果目标文件已存在，跳过（避免覆盖）
                if os.path.exists(target_file):
                    continue
                
                # 复制文件
                try:
                    shutil.copy2(source_file, target_file)
                    stats[split][subdir] += 1
                except Exception as e:
                    print(f"  警告：复制文件失败 {filename}: {e}")
            
            if stats[split][subdir] > 0:
                print(f"  {subdir}: 复制了 {stats[split][subdir]} 个文件")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("整合完成！")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        total = sum(stats[split].values())
        if total > 0:
            print(f"\n{split} 集:")
            for subdir in ['video', 'audio', 'text', 'labels']:
                count = stats[split][subdir]
                if count > 0:
                    print(f"  {subdir}: {count} 个文件")
    
    # 验证整合结果
    print("\n验证整合结果...")
    for split in ['train', 'val', 'test']:
        video_dir = os.path.join(TARGET_ROOT, split, 'video')
        if os.path.exists(video_dir):
            mosei_count = len([f for f in os.listdir(video_dir) if f.startswith('mosei_')])
            total_count = len([f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))])
            print(f"{split}: 总计 {total_count} 个视频文件（其中 MOSEI: {mosei_count} 个）")
    
    print(f"\n数据已整合到: {TARGET_ROOT}")
    print("现在所有数据集都在统一的 data/train/val/test/ 目录下")


def main():
    try:
        merge_dataset()
    except Exception as e:
        print("\n错误：整合过程中出现异常：")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
移动CREMA-D数据集到目标路径的脚本
"""

import os
import shutil
import sys

# 源路径（当前数据集位置）
SOURCE_PATH = "/home/lizhichun_24/.cache/kagglehub/datasets/orvile/crema-d-emotional-multimodal-dataset/versions/1"

# 目标路径
TARGET_PATH = "/home/lizhichun_24/sda1/code/multimodal/project/downloads"

def move_dataset(source, target):
    """
    移动数据集从源路径到目标路径
    
    Args:
        source: 源路径
        target: 目标路径
    """
    print("=" * 60)
    print("CREMA-D数据集移动脚本")
    print("=" * 60)
    print(f"源路径: {source}")
    print(f"目标路径: {target}")
    print("=" * 60)
    
    # 检查源路径是否存在
    if not os.path.exists(source):
        print(f"错误：源路径不存在: {source}")
        return False
    
    # 确保目标目录存在
    os.makedirs(target, exist_ok=True)
    
    # 获取数据集名称
    dataset_name = "crema-d-emotional-multimodal-dataset"
    final_path = os.path.join(target, dataset_name)
    
    # 如果目标路径已存在，询问是否覆盖
    if os.path.exists(final_path):
        print(f"警告：目标路径已存在: {final_path}")
        response = input("是否覆盖？(y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return False
        # 删除已存在的目录
        shutil.rmtree(final_path)
    
    # 移动数据集
    print(f"\n正在移动数据集...")
    print(f"从: {source}")
    print(f"到: {final_path}")
    
    try:
        shutil.move(source, final_path)
        print(f"\n✓ 移动完成！")
        print(f"数据集位置: {final_path}")
        
        # 验证移动结果
        if os.path.exists(final_path):
            # 计算大小
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(final_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                        file_count += 1
            
            print(f"\n数据集信息:")
            print(f"  文件数量: {file_count}")
            print(f"  总大小: {total_size / (1024**3):.2f} GB")
            
        return True
        
    except Exception as e:
        print(f"\n错误：移动失败: {e}")
        return False

if __name__ == "__main__":
    # 如果提供了命令行参数，使用参数作为路径
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = TARGET_PATH
    
    # 如果提供了两个参数，第一个是源路径，第二个是目标路径
    if len(sys.argv) > 2:
        source_path = sys.argv[1]
        target_path = sys.argv[2]
    else:
        source_path = SOURCE_PATH
    
    success = move_dataset(source_path, target_path)
    
    if success:
        print("\n" + "=" * 60)
        print("移动完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("移动失败！")
        print("=" * 60)
        sys.exit(1)


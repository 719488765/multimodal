#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CREMA-D数据集下载脚本
使用kagglehub下载到指定路径

目标下载路径：/home/lizhichun_24/sda1/code/multimodal/project/downloads

使用方法：
    python scripts/download_crema_d.py
    或
    python scripts/download_crema_d.py [自定义路径]
"""

import os
import sys
import shutil

# 默认目标路径
DEFAULT_TARGET_PATH = "/home/lizhichun_24/sda1/code/multimodal/project/downloads"

# 尝试导入kagglehub，如果失败给出安装提示
try:
    import kagglehub
except ImportError:
    print("=" * 60)
    print("错误：未找到 kagglehub 库")
    print("=" * 60)
    print("请先安装 kagglehub：")
    print("  pip install kagglehub")
    print("\n或者在远程服务器上执行：")
    print("  conda activate myenv310")
    print("  pip install kagglehub")
    print("=" * 60)
    sys.exit(1)

def download_crema_d(target_path):
    """
    下载CREMA-D数据集到指定路径
    
    Args:
        target_path: 目标下载路径
    """
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)
    print(f"目标路径: {target_path}")
    
    try:
        # 方法1：尝试使用path参数（如果kagglehub支持）
        # 注意：需要检查kagglehub版本是否支持path参数
        print("开始下载CREMA-D数据集...")
        print("数据集: orvile/crema-d-emotional-multimodal-dataset")
        
        # 下载数据集
        # 如果kagglehub支持path参数：
        path = kagglehub.dataset_download(
            "orvile/crema-d-emotional-multimodal-dataset",
            path=target_path  # 指定下载位置
        )
        
        print(f"✓ 下载完成！")
        print(f"数据集路径: {path}")
        return path
        
    except TypeError as e:
        # 如果path参数不支持，使用环境变量方法
        print(f"path参数不支持，使用环境变量方法: {e}")
        return download_with_env_var(target_path)
    except Exception as e:
        print(f"下载出错: {e}")
        print("尝试使用环境变量方法...")
        return download_with_env_var(target_path)

def download_with_env_var(target_path):
    """
    使用环境变量指定下载路径
    """
    import os
    
    # 设置kagglehub缓存目录
    os.environ['KAGGLEHUB_HOME'] = target_path
    
    print(f"设置KAGGLEHUB_HOME={target_path}")
    
    # 下载数据集
    path = kagglehub.dataset_download("orvile/crema-d-emotional-multimodal-dataset")
    
    print(f"✓ 下载完成！")
    print(f"数据集路径: {path}")
    return path

def download_and_move(target_path):
    """
    下载到默认位置后移动到目标路径
    """
    import shutil
    
    # 先下载到默认位置
    print("下载到默认位置...")
    default_path = kagglehub.dataset_download("orvile/crema-d-emotional-multimodal-dataset")
    
    print(f"默认下载路径: {default_path}")
    
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)
    
    # 移动文件
    dataset_name = os.path.basename(default_path)
    final_path = os.path.join(target_path, dataset_name)
    
    if os.path.exists(final_path):
        print(f"目标路径已存在: {final_path}")
        print("跳过移动操作")
        return final_path
    
    print(f"移动数据集到: {final_path}")
    shutil.move(default_path, final_path)
    
    print(f"✓ 移动完成！")
    return final_path

if __name__ == "__main__":
    # 目标路径：如果提供了命令行参数，使用参数作为路径；否则使用默认路径
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        print(f"使用命令行指定的路径: {target_path}")
    else:
        target_path = DEFAULT_TARGET_PATH
        print(f"使用默认路径: {target_path}")
    
    print("=" * 60)
    print("CREMA-D数据集下载脚本")
    print("=" * 60)
    print(f"目标下载路径: {target_path}")
    print("=" * 60)
    
    # 检查目标路径的父目录是否存在
    parent_dir = os.path.dirname(target_path)
    if not os.path.exists(parent_dir):
        print(f"警告：父目录不存在: {parent_dir}")
        print("正在创建父目录...")
        os.makedirs(parent_dir, exist_ok=True)
    
    # 尝试方法1：使用path参数
    try:
        path = download_crema_d(target_path)
    except Exception as e:
        print(f"方法1失败: {e}")
        # 尝试方法3：下载后移动
        print("\n尝试方法3：下载后移动...")
        path = download_and_move(target_path)
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print(f"数据集位置: {path}")
    print("=" * 60)

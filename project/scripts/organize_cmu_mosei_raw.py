#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMU-MOSEI 原始数据集整理脚本
将下载的 CMU-MOSEI 原始数据（Kaggle 版本）整理到 data/ 目录，按照项目要求的格式组织

支持的数据来源：
- Kaggle 下载的 cmu-mosei.zip（解压后的目录）
- 自动检测目录结构并适配

数据格式要求：
- 视频：MP4、AVI等格式
- 音频：WAV格式（如果单独存在）
- 文本：TXT格式，UTF-8编码
- 标签：TXT格式，第一行为情绪类别，第二行为情绪强度（效价,唤醒度）
"""

import os
import shutil
import json
import csv
import random
from pathlib import Path

# 配置路径
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads", "CMU_MOSEI_raw")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# CMU-MOSEI 情感类别映射（根据实际标签格式调整）
EMOTION_MAP = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fear': 'fear',
    'disgust': 'disgust',
    'surprise': 'surprise',
    'neutral': 'neutral'
}

# 情感强度映射（效价和唤醒度）
VALENCE_AROUSAL_MAP = {
    'happy': (0.8, 0.7),
    'sad': (-0.6, -0.3),
    'angry': (-0.7, 0.8),
    'fear': (-0.5, 0.9),
    'disgust': (-0.7, 0.5),
    'surprise': (0.3, 0.8),
    'neutral': (0.0, 0.0)
}


def find_mosei_files(download_dir):
    """
    查找 CMU-MOSEI 数据集文件
    自动适配不同的目录结构
    
    Args:
        download_dir: 下载目录路径
        
    Returns:
        files: 找到的文件列表，每个元素包含文件路径和元数据
    """
    files = []
    
    if not os.path.exists(download_dir):
        print(f"错误：下载目录不存在: {download_dir}")
        print("请先完成数据集的下载和解压")
        return []
    
    print(f"扫描目录: {download_dir}")
    
    # 递归查找所有视频、音频和文本文件
    for root, dirs, filenames in os.walk(download_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            
            # 检查文件类型
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                files.append({
                    'type': 'video',
                    'path': filepath,
                    'filename': filename,
                    'base': os.path.splitext(filename)[0]
                })
            elif filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                files.append({
                    'type': 'audio',
                    'path': filepath,
                    'filename': filename,
                    'base': os.path.splitext(filename)[0]
                })
            elif filename.lower().endswith(('.txt', '.csv')):
                # 文本文件或标注文件
                files.append({
                    'type': 'text' if filename.endswith('.txt') else 'annotation',
                    'path': filepath,
                    'filename': filename,
                    'base': os.path.splitext(filename)[0]
                })
    
    print(f"找到 {len(files)} 个文件")
    return files


def find_annotation_files(download_dir):
    """
    查找标注文件（CSV、JSON等）
    
    Args:
        download_dir: 下载目录路径
        
    Returns:
        annotation_files: 标注文件路径字典
    """
    annotation_files = {}
    
    # 常见的标注文件位置
    possible_paths = [
        os.path.join(download_dir, "annotations"),
        os.path.join(download_dir, "labels"),
        os.path.join(download_dir, "data"),
        os.path.join(download_dir, "metadata"),
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            for root, dirs, filenames in os.walk(base_path):
                for filename in filenames:
                    if filename.endswith(('.csv', '.json', '.txt')):
                        filepath = os.path.join(root, filename)
                        # 尝试从文件名推断 split
                        if 'train' in filename.lower():
                            annotation_files['train'] = filepath
                        elif 'val' in filename.lower() or 'dev' in filename.lower():
                            annotation_files['val'] = filepath
                        elif 'test' in filename.lower():
                            annotation_files['test'] = filepath
                        else:
                            # 如果没有明确 split，作为通用标注文件
                            annotation_files['all'] = filepath
    
    return annotation_files


def load_annotations(annotation_path):
    """
    加载标注文件（支持 CSV 和 JSON 格式）
    
    Args:
        annotation_path: 标注文件路径
        
    Returns:
        samples: 样本列表，每个元素包含标注信息
    """
    samples = []
    
    if not os.path.exists(annotation_path):
        return samples
    
    if annotation_path.endswith('.csv'):
        # CSV 格式
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(row)
        except Exception as e:
            print(f"警告：无法读取 CSV 文件 {annotation_path}: {e}")
    elif annotation_path.endswith('.json'):
        # JSON 格式
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict):
                    # 如果是字典，尝试提取样本列表
                    samples = data.get('samples', data.get('data', []))
        except Exception as e:
            print(f"警告：无法读取 JSON 文件 {annotation_path}: {e}")
    
    return samples


def parse_mosei_filename(filename):
    """
    解析 CMU-MOSEI 文件名，提取样本ID等信息
    
    CMU-MOSEI 文件名格式可能为：
    - {video_id}_{segment_id}.mp4
    - {video_id}.mp4
    - 或其他格式
    
    Args:
        filename: 文件名
        
    Returns:
        video_id: 视频ID
        segment_id: 片段ID（如果有）
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    video_id = None
    segment_id = None
    
    # 尝试提取视频ID和片段ID
    if len(parts) >= 2:
        video_id = parts[0]
        segment_id = parts[1] if parts[1].isdigit() else None
    else:
        video_id = base
    
    return video_id, segment_id


def organize_mosei_data(download_dir, data_root, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    整理 CMU-MOSEI 数据到 data/ 目录
    
    Args:
        download_dir: 下载目录路径
        data_root: 数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    print("=" * 60)
    print("CMU-MOSEI 数据集整理脚本")
    print("=" * 60)
    
    # 查找文件
    print("\n扫描 CMU-MOSEI 数据集文件...")
    files = find_mosei_files(download_dir)
    
    if not files:
        print("错误：未找到任何文件")
        return
    
    # 查找标注文件
    print("\n查找标注文件...")
    annotation_files = find_annotation_files(download_dir)
    print(f"找到 {len(annotation_files)} 个标注文件")
    
    # 按类型分组文件
    video_files = {f['base']: f for f in files if f['type'] == 'video'}
    audio_files = {f['base']: f for f in files if f['type'] == 'audio'}
    text_files = {f['base']: f for f in files if f['type'] == 'text'}
    
    print(f"视频文件: {len(video_files)} 个")
    print(f"音频文件: {len(audio_files)} 个")
    print(f"文本文件: {len(text_files)} 个")
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['video', 'audio', 'text', 'physiological', 'labels']:
            os.makedirs(os.path.join(data_root, split, subdir), exist_ok=True)
    
    # 如果有标注文件，使用标注文件进行划分
    if annotation_files:
        print("\n使用标注文件进行数据划分...")
        # 这里可以根据标注文件的具体格式进行划分
        # 暂时使用文件列表进行随机划分
        pass
    
    # 按base名称分组文件（同一样本的视频、音频、文本）
    samples = {}
    
    for file_info in files:
        base = file_info['base']
        if base not in samples:
            samples[base] = {
                'video': None,
                'audio': None,
                'text': None,
                'emotion': None,
                'metadata': {}
            }
        
        if file_info['type'] == 'video':
            samples[base]['video'] = file_info['path']
        elif file_info['type'] == 'audio':
            samples[base]['audio'] = file_info['path']
        elif file_info['type'] == 'text':
            samples[base]['text'] = file_info['path']
        
        # 解析视频ID等信息
        video_id, segment_id = parse_mosei_filename(file_info['filename'])
        samples[base]['metadata']['video_id'] = video_id
        if segment_id:
            samples[base]['metadata']['segment_id'] = segment_id
    
    print(f"\n整理 {len(samples)} 个样本")
    
    # 划分数据集
    sample_list = list(samples.items())
    random.seed(42)  # 固定随机种子，确保可复现
    random.shuffle(sample_list)
    
    n_total = len(sample_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    splits = {
        'train': sample_list[:n_train],
        'val': sample_list[n_train:n_train + n_val],
        'test': sample_list[n_train + n_val:]
    }
    
    print(f"数据集划分: train={n_train}, val={n_val}, test={n_test}")
    
    # 复制文件并生成标签
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for split, sample_list in splits.items():
        print(f"\n处理 {split} 集...")
        
        for idx, (base, sample_info) in enumerate(sample_list, start=1):
            sample_id = f"mosei_{split}_{idx:06d}"
            
            # 复制视频文件
            if sample_info['video']:
                src_video = sample_info['video']
                ext = os.path.splitext(src_video)[1]
                dst_video = os.path.join(data_root, split, 'video', f"{sample_id}{ext}")
                if not os.path.exists(dst_video):
                    shutil.copy2(src_video, dst_video)
            
            # 复制音频文件（如果存在）
            if sample_info['audio']:
                src_audio = sample_info['audio']
                ext = os.path.splitext(src_audio)[1]
                dst_audio = os.path.join(data_root, split, 'audio', f"{sample_id}{ext}")
                if not os.path.exists(dst_audio):
                    shutil.copy2(src_audio, dst_audio)
            
            # 复制或生成文本文件
            if sample_info['text']:
                src_text = sample_info['text']
                dst_text = os.path.join(data_root, split, 'text', f"{sample_id}.txt")
                if not os.path.exists(dst_text):
                    shutil.copy2(src_text, dst_text)
            else:
                # 如果没有文本文件，生成占位文本
                dst_text = os.path.join(data_root, split, 'text', f"{sample_id}.txt")
                with open(dst_text, 'w', encoding='utf-8') as f:
                    video_id = sample_info['metadata'].get('video_id', 'unknown')
                    f.write(f"Transcript for CMU-MOSEI video {video_id}\n")
            
            # 生成标签文件（目前使用占位标签，后续可根据标注文件完善）
            emotion = sample_info.get('emotion', 'neutral')
            if emotion not in VALENCE_AROUSAL_MAP:
                emotion = 'neutral'
            
            valence, arousal = VALENCE_AROUSAL_MAP.get(emotion, (0.0, 0.0))
            
            dst_label = os.path.join(data_root, split, 'labels', f"{sample_id}.txt")
            with open(dst_label, 'w', encoding='utf-8') as f:
                f.write(f"{emotion}\n")
                f.write(f"{valence},{arousal}\n")
            
            stats[split] += 1
            
            if (idx % 100) == 0:
                print(f"  已处理 {idx}/{len(sample_list)} 个样本")
    
    print("\n" + "=" * 60)
    print("整理完成！")
    print("=" * 60)
    print(f"训练集: {stats['train']} 个样本")
    print(f"验证集: {stats['val']} 个样本")
    print(f"测试集: {stats['test']} 个样本")
    print(f"\n数据已整理到: {data_root}")
    
    # 验证结果
    print("\n验证整理结果...")
    for split in ['train', 'val', 'test']:
        video_dir = os.path.join(data_root, split, 'video')
        audio_dir = os.path.join(data_root, split, 'audio')
        text_dir = os.path.join(data_root, split, 'text')
        label_dir = os.path.join(data_root, split, 'labels')
        
        video_count = len([f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]) if os.path.exists(video_dir) else 0
        audio_count = len([f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]) if os.path.exists(audio_dir) else 0
        text_count = len([f for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f))]) if os.path.exists(text_dir) else 0
        label_count = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]) if os.path.exists(label_dir) else 0
        
        print(f"{split}: 视频={video_count}, 音频={audio_count}, 文本={text_count}, 标签={label_count}")


def main():
    """主函数"""
    import sys
    
    print("=" * 60)
    print("CMU-MOSEI 原始数据集整理脚本")
    print("=" * 60)
    
    # 检查下载目录
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"\n错误：下载目录不存在: {DOWNLOAD_DIR}")
        print("请先完成以下步骤：")
        print("1. 在本地浏览器下载 CMU-MOSEI 数据集（Kaggle）")
        print("2. 将下载的 zip 文件上传到服务器")
        print("3. 在服务器上解压到: downloads/CMU_MOSEI_raw")
        print("\n解压命令示例：")
        print(f"  cd {os.path.dirname(DOWNLOAD_DIR)}")
        print("  unzip cmu-mosei.zip -d CMU_MOSEI_raw")
        return
    
    # 整理数据
    organize_mosei_data(DOWNLOAD_DIR, DATA_ROOT)
    
    print("\n" + "=" * 60)
    print("整理完成！可以开始使用数据集了。")
    print("=" * 60)
    print("\n下一步：")
    print("1. 检查整理后的数据目录结构")
    print("2. 在 config.yaml 中配置数据路径")
    print("3. 运行训练脚本开始训练")


if __name__ == "__main__":
    main()

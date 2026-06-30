#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CREMA-D数据集整理脚本
将下载的CREMA-D数据集整理到data/目录，按照项目要求的格式组织
"""

import os
import shutil
import json
import random
from pathlib import Path

# 配置路径
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads", "crema-d-emotional-multimodal-dataset")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# CREMA-D情感类别映射
EMOTION_MAP = {
    'HAP': 'happy',      # Happy
    'SAD': 'sad',        # Sad
    'ANG': 'angry',      # Angry
    'FEA': 'fear',       # Fear
    'DIS': 'disgust',    # Disgust
    'NEU': 'neutral'     # Neutral
}

def find_crema_d_files(download_dir):
    """
    查找CREMA-D数据集文件
    
    Args:
        download_dir: 下载目录路径
        
    Returns:
        files: 找到的文件列表，每个元素包含文件路径和元数据
    """
    files = []
    
    # 如果目录不存在，尝试查找可能的路径
    if not os.path.exists(download_dir):
        # 尝试查找可能的路径（包括content/CREMA-D结构）
        possible_paths = [
            os.path.join(PROJECT_ROOT, "downloads", "crema-d-emotional-multimodal-dataset", "content", "CREMA-D"),
            os.path.join(PROJECT_ROOT, "downloads", "crema-d-emotional-multimodal-dataset"),
            os.path.join(PROJECT_ROOT, "downloads", "CREMA-D"),
            os.path.join(PROJECT_ROOT, "downloads"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                download_dir = path
                print(f"找到数据集目录: {download_dir}")
                break
        else:
            print(f"错误：找不到CREMA-D数据集目录")
            print(f"请检查数据集是否在以下位置之一：")
            for path in possible_paths:
                print(f"  - {path}")
            return []
    
    print(f"扫描目录: {download_dir}")
    
    # 递归查找所有视频和音频文件
    for root, dirs, filenames in os.walk(download_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            
            # 检查文件类型（添加.flv支持）
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                # 视频文件
                files.append({
                    'type': 'video',
                    'path': filepath,
                    'filename': filename,
                    'base': os.path.splitext(filename)[0]
                })
            elif filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                # 音频文件（如果用户需要处理音频）
                files.append({
                    'type': 'audio',
                    'path': filepath,
                    'filename': filename,
                    'base': os.path.splitext(filename)[0]
                })
    
    print(f"找到 {len(files)} 个文件")
    return files

def parse_crema_d_filename(filename):
    """
    解析CREMA-D文件名，提取情感标签
    
    CREMA-D文件名格式通常为：
    - 1001_DFA_ANG_XX.wav (ActorID_Emotion_Intensity_Statement)
    - 或类似格式
    
    Args:
        filename: 文件名
        
    Returns:
        emotion: 情感类别
        metadata: 其他元数据
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    emotion = None
    intensity = None
    actor_id = None
    
    # 尝试从文件名提取情感标签
    for part in parts:
        part_upper = part.upper()
        if part_upper in EMOTION_MAP:
            emotion = EMOTION_MAP[part_upper]
            break
    
    # 如果找不到，尝试其他可能的格式
    if emotion is None:
        # 尝试查找情感关键词
        base_upper = base.upper()
        for key, value in EMOTION_MAP.items():
            if key in base_upper:
                emotion = value
                break
    
    # 如果还是找不到，使用neutral作为默认值
    if emotion is None:
        emotion = 'neutral'
        print(f"警告：无法从文件名 '{filename}' 提取情感标签，使用 'neutral'")
    
    # 提取强度（如果有）
    for part in parts:
        if part in ['LO', 'MD', 'HI', 'XX']:
            intensity = part
            break
    
    # 提取演员ID（通常是第一个数字部分）
    for part in parts:
        if part.isdigit():
            actor_id = part
            break
    
    return emotion, {
        'intensity': intensity,
        'actor_id': actor_id,
        'original_filename': filename
    }

def load_audio_mapping(data_root):
    """
    加载已存在的音频文件映射
    如果存在映射文件，则使用映射文件；否则通过索引位置匹配
    
    Args:
        data_root: 数据根目录
        
    Returns:
        audio_mapping: 音频文件映射，格式为 {(split, idx): audio_filename}
    """
    audio_mapping = {}
    
    # 检查是否存在映射文件
    mapping_file = os.path.join(data_root, 'audio_file_mapping.json')
    if os.path.exists(mapping_file):
        print(f"  发现映射文件: {mapping_file}")
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                # 转换格式
                for key, value in mapping_data.items():
                    split, idx = key.split('_')
                    audio_mapping[(split, int(idx))] = value
            print(f"  从映射文件加载 {len(audio_mapping)} 个映射")
            return audio_mapping
        except Exception as e:
            print(f"  警告：无法读取映射文件: {e}")
    
    # 如果没有映射文件，通过索引位置匹配
    print("  通过索引位置匹配音频文件...")
    for split in ['train', 'val', 'test']:
        audio_dir = os.path.join(data_root, split, 'audio')
        if os.path.exists(audio_dir):
            audio_files = sorted([f for f in os.listdir(audio_dir) 
                                if os.path.isfile(os.path.join(audio_dir, f))])
            for idx, audio_file in enumerate(audio_files, start=1):
                audio_mapping[(split, idx)] = audio_file
    
    return audio_mapping

def organize_files(files, data_root, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, skip_audio=False):
    """
    整理文件到data/目录
    
    Args:
        files: 文件列表
        data_root: 数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        skip_audio: 是否跳过音频处理（如果音频已整理）
    """
    # 检查是否已有音频文件
    existing_audio = {}
    if skip_audio:
        print("检测到跳过音频选项，检查已存在的音频文件...")
        # 检查data目录中是否已有音频文件
        for split in ['train', 'val', 'test']:
            audio_dir = os.path.join(data_root, split, 'audio')
            if os.path.exists(audio_dir) and os.listdir(audio_dir):
                print(f"  发现 {split} 集已有 {len(os.listdir(audio_dir))} 个音频文件，将跳过音频处理")
                skip_audio = True
                break
    
    # 只处理视频文件（如果skip_audio为True）
    if skip_audio:
        files = [f for f in files if f['type'] == 'video']
        print(f"只处理视频文件，共 {len(files)} 个")
    
    # 按base名称分组文件（同一样本的视频和音频）
    samples = {}
    
    for file_info in files:
        base = file_info['base']
        if base not in samples:
            samples[base] = {
                'video': None,
                'audio': None,
                'emotion': None,
                'metadata': {}
            }
        
        if file_info['type'] == 'video':
            samples[base]['video'] = file_info['path']
        elif file_info['type'] == 'audio' and not skip_audio:
            samples[base]['audio'] = file_info['path']
        
        # 解析情感标签（从第一个文件解析即可）
        if samples[base]['emotion'] is None:
            emotion, metadata = parse_crema_d_filename(file_info['filename'])
            samples[base]['emotion'] = emotion
            samples[base]['metadata'] = metadata
    
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
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['video', 'audio', 'text', 'physiological', 'labels']:
            os.makedirs(os.path.join(data_root, split, subdir), exist_ok=True)
    
    # 如果跳过音频，预先加载已存在的音频文件映射
    audio_mapping = {}  # key: (split, idx), value: audio_filename
    if skip_audio:
        print("\n扫描已存在的音频文件，建立匹配索引...")
        audio_mapping = load_audio_mapping(data_root)
        if audio_mapping:
            print(f"  已建立 {len(audio_mapping)} 个音频文件映射")
    
    # 复制文件并生成标签
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for split, sample_list in splits.items():
        print(f"\n处理 {split} 集...")
        
        for idx, (base, sample_info) in enumerate(sample_list, start=1):
            sample_id = f"crema_{split}_{idx:04d}"
            
            # 复制视频文件
            if sample_info['video']:
                src_video = sample_info['video']
                # 确定目标扩展名（保持原始格式，如.flv）
                ext = os.path.splitext(src_video)[1]
                dst_video = os.path.join(data_root, split, 'video', f"{sample_id}{ext}")
                os.makedirs(os.path.dirname(dst_video), exist_ok=True)
                if not os.path.exists(dst_video):
                    shutil.copy2(src_video, dst_video)
                else:
                    print(f"  跳过已存在的视频: {sample_id}{ext}")
            
            # 复制音频文件（如果未跳过且存在）
            if not skip_audio and sample_info['audio']:
                src_audio = sample_info['audio']
                ext = os.path.splitext(src_audio)[1]
                dst_audio = os.path.join(data_root, split, 'audio', f"{sample_id}{ext}")
                os.makedirs(os.path.dirname(dst_audio), exist_ok=True)
                if not os.path.exists(dst_audio):
                    shutil.copy2(src_audio, dst_audio)
                else:
                    print(f"  跳过已存在的音频: {sample_id}{ext}")
            elif skip_audio:
                # 如果跳过音频，检查是否已有对应的音频文件（通过索引匹配）
                if (split, idx) in audio_mapping:
                    matching_audio = audio_mapping[(split, idx)]
                    # 验证文件确实存在
                    audio_dir = os.path.join(data_root, split, 'audio')
                    matching_audio_path = os.path.join(audio_dir, matching_audio)
                    if os.path.exists(matching_audio_path):
                        # 音频文件已存在，跳过处理
                        if (idx % 100) == 0:
                            print(f"  已匹配音频: {matching_audio}")
                    else:
                        # 索引存在但文件不存在，记录警告
                        print(f"  警告：映射的音频文件不存在: {matching_audio}")
            
            # 生成文本文件（占位，CREMA-D通常没有文本）
            dst_text = os.path.join(data_root, split, 'text', f"{sample_id}.txt")
            with open(dst_text, 'w', encoding='utf-8') as f:
                # 可以从音频转文本，这里使用占位文本
                f.write(f"Audio transcription for {sample_id}\n")
            
            # 生成标签文件
            emotion = sample_info['emotion']
            # 默认情绪强度（效价和唤醒度）
            # 这里使用简化的映射，实际应该根据CREMA-D的标注
            valence_arousal_map = {
                'happy': (0.8, 0.7),
                'sad': (-0.6, -0.3),
                'angry': (-0.7, 0.8),
                'fear': (-0.5, 0.9),
                'disgust': (-0.7, 0.5),
                'neutral': (0.0, 0.0)
            }
            
            valence, arousal = valence_arousal_map.get(emotion, (0.0, 0.0))
            
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
    
    return stats

def main():
    """主函数"""
    import sys
    
    print("=" * 60)
    print("CREMA-D数据集整理脚本")
    print("=" * 60)
    
    # 检查命令行参数
    skip_audio = '--skip-audio' in sys.argv or '-s' in sys.argv
    
    if skip_audio:
        print("跳过音频处理模式：只整理视频文件")
    
    # 检查下载目录
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"\n警告：下载目录不存在: {DOWNLOAD_DIR}")
        print("尝试查找数据集...")
    
    # 查找文件
    files = find_crema_d_files(DOWNLOAD_DIR)
    
    if not files:
        print("\n错误：未找到任何文件，请检查：")
        print(f"1. 数据集是否已下载到: {DOWNLOAD_DIR}")
        print(f"2. 数据集目录结构是否正确")
        print("\n提示：可以手动检查数据集目录：")
        print(f"  ls -la {DOWNLOAD_DIR}")
        print(f"  find {DOWNLOAD_DIR} -name '*.flv' | head -10")
        return
    
    # 如果只处理视频，过滤文件列表
    if skip_audio:
        video_files = [f for f in files if f['type'] == 'video']
        print(f"\n只处理视频文件: {len(video_files)} 个")
        if len(video_files) == 0:
            print("错误：未找到视频文件（.flv, .mp4, .avi等）")
            print("请检查数据集目录中是否包含视频文件")
            return
        files = video_files
    
    # 检查是否已有音频文件（自动检测）
    auto_skip_audio = False
    for split in ['train', 'val', 'test']:
        audio_dir = os.path.join(DATA_ROOT, split, 'audio')
        if os.path.exists(audio_dir) and len(os.listdir(audio_dir)) > 0:
            auto_skip_audio = True
            print(f"\n检测到 {split} 集已有音频文件，自动跳过音频处理")
            break
    
    # 整理文件
    stats = organize_files(files, DATA_ROOT, skip_audio=(skip_audio or auto_skip_audio))
    
    # 验证结果
    print("\n验证整理结果...")
    for split in ['train', 'val', 'test']:
        video_dir = os.path.join(DATA_ROOT, split, 'video')
        audio_dir = os.path.join(DATA_ROOT, split, 'audio')
        label_dir = os.path.join(DATA_ROOT, split, 'labels')
        
        video_count = len([f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]) if os.path.exists(video_dir) else 0
        audio_count = len([f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]) if os.path.exists(audio_dir) else 0
        label_count = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]) if os.path.exists(label_dir) else 0
        
        print(f"{split}: 视频={video_count}, 音频={audio_count}, 标签={label_count}")
    
    print("\n" + "=" * 60)
    print("整理完成！可以开始使用数据集了。")
    print("=" * 60)

if __name__ == "__main__":
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MELD数据集整理脚本
将下载的MELD数据集整理到data/目录，按照项目要求的格式组织
"""

import os
import shutil
import csv
import random
import sys

# 配置路径
# 为了兼容不同环境，项目根目录使用相对当前脚本的位置自动推断
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.meld_audio_utils import extract_audio_from_video, meld_wav_path_for_video
MELD_ROOT = os.path.join(PROJECT_ROOT, "downloads", "MELD")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# MELD情感类别映射到项目格式
EMOTION_MAP = {
    'neutral': 'neutral',
    'joy': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'surprise': 'surprise',
    'fear': 'fear',
    'disgust': 'disgust'
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

def find_meld_files(meld_root):
    """
    查找MELD数据集的视频文件和标注文件
    
    Args:
        meld_root: MELD数据集根目录
        
    Returns:
        video_files: 视频文件字典，key为(dialogue_id, utterance_id)，value为文件路径
        csv_files: CSV标注文件路径字典
    """
    video_files = {}
    csv_files = {}
    
    # 查找视频文件
    video_base_dir = os.path.join(meld_root, "videos")
    if os.path.exists(video_base_dir):
        # 支持两种目录结构：
        # 1）标准结构：videos/train, videos/dev, videos/test
        # 2）MELD原始结构：videos/train_splits, videos/dev_splits_complete, videos/output_repeated_splits_test
        split_dir_mapping = {
            'train': ['train', 'train_splits'],
            'dev': ['dev', 'dev_splits_complete'],
            'test': ['test', 'output_repeated_splits_test'],
        }

        for split, candidate_dirs in split_dir_mapping.items():
            for subdir in candidate_dirs:
                video_dir = os.path.join(video_base_dir, subdir)
                if not os.path.exists(video_dir):
                    continue

                # 递归遍历所有子目录，查找 .mp4 文件
                for root, _, files in os.walk(video_dir):
                    for filename in files:
                        if not filename.endswith('.mp4'):
                            continue
                        # 解析文件名：dia{dialogue_id}_utt{utterance_id}.mp4
                        if filename.startswith('dia') and '_utt' in filename:
                            try:
                                parts = filename.replace('.mp4', '').split('_')
                                dialogue_id = parts[0].replace('dia', '')
                                utterance_id = parts[1].replace('utt', '')
                                key = (split, dialogue_id, utterance_id)
                                video_files[key] = os.path.join(root, filename)
                            except Exception:
                                print(f"警告：无法解析文件名: {filename}")
    
    # 查找CSV标注文件
    csv_base_dir = os.path.join(meld_root, "data", "MELD")
    if os.path.exists(csv_base_dir):
        csv_files['train'] = os.path.join(csv_base_dir, "train_sent_emo.csv")
        csv_files['dev'] = os.path.join(csv_base_dir, "dev_sent_emo.csv")
        csv_files['test'] = os.path.join(csv_base_dir, "test_sent_emo.csv")
    
    return video_files, csv_files

def load_csv_annotations(csv_path):
    """
    加载CSV标注文件
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        samples: 样本列表，每个元素包含标注信息
    """
    samples = []
    
    if not os.path.exists(csv_path):
        print(f"警告：CSV文件不存在: {csv_path}")
        return samples
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'dialogue_id': row.get('Dialogue_ID', ''),
                'utterance_id': row.get('Utterance_ID', ''),
                'emotion': row.get('Emotion', 'neutral').lower(),
                'sentiment': row.get('Sentiment', 'neutral').lower(),
                'utterance': row.get('Utterance', ''),
                'speaker': row.get('Speaker', '')
            })
    
    return samples

def organize_meld_data(meld_root, data_root):
    """
    整理MELD数据到data/目录
    
    Args:
        meld_root: MELD数据集根目录
        data_root: 数据根目录
    """
    print("=" * 60)
    print("MELD数据集整理脚本")
    print("=" * 60)
    
    # 查找文件
    print("\n扫描MELD数据集文件...")
    video_files, csv_files = find_meld_files(meld_root)
    
    print(f"找到视频文件: {len(video_files)} 个")
    print(f"找到CSV文件: {len(csv_files)} 个")
    
    if not video_files:
        print("错误：未找到视频文件")
        print(f"请检查数据集是否在: {meld_root}")
        return
    
    if not csv_files:
        print("错误：未找到CSV标注文件")
        print(f"请检查标注文件是否在: {os.path.join(meld_root, 'data', 'MELD')}")
        return
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['video', 'audio', 'text', 'physiological', 'labels']:
            os.makedirs(os.path.join(data_root, split, subdir), exist_ok=True)
    
    # 处理每个数据集划分
    split_mapping = {
        'train': 'train',
        'dev': 'val',  # MELD的dev映射到项目的val
        'test': 'test'
    }
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for meld_split, project_split in split_mapping.items():
        print(f"\n处理 {meld_split} 集（映射到 {project_split}）...")
        
        csv_path = csv_files.get(meld_split)
        if not csv_path or not os.path.exists(csv_path):
            print(f"  跳过：CSV文件不存在")
            continue
        
        # 加载标注
        samples = load_csv_annotations(csv_path)
        print(f"  加载 {len(samples)} 个标注样本")
        
        # 处理每个样本
        valid_count = 0
        for idx, sample in enumerate(samples, start=1):
            dialogue_id = sample['dialogue_id']
            utterance_id = sample['utterance_id']
            
            # 查找对应的视频文件
            video_key = (meld_split, dialogue_id, utterance_id)
            if video_key not in video_files:
                # 尝试其他可能的格式
                video_key = (meld_split, str(int(dialogue_id)), str(int(utterance_id)))
                if video_key not in video_files:
                    continue
            
            video_path = video_files[video_key]
            if not os.path.exists(video_path):
                continue
            
            sample_id = f"meld_{project_split}_{idx:04d}"
            
            # 复制视频文件
            dst_video = os.path.join(data_root, project_split, 'video', f"{sample_id}.mp4")
            if not os.path.exists(dst_video):
                shutil.copy2(video_path, dst_video)

            # 从 mp4 提取 mono 16kHz WAV（与 config data.audio.sample_rate 一致）
            dst_audio = meld_wav_path_for_video(dst_video)
            ok, msg = extract_audio_from_video(dst_video, dst_audio)
            if not ok and msg != "skipped existing":
                print(f"  警告：音频提取失败 {sample_id}: {msg}")
            
            # 生成文本文件
            dst_text = os.path.join(data_root, project_split, 'text', f"{sample_id}.txt")
            with open(dst_text, 'w', encoding='utf-8') as f:
                f.write(sample['utterance'] + "\n")
            
            # 生成标签文件
            emotion = EMOTION_MAP.get(sample['emotion'], 'neutral')
            valence, arousal = VALENCE_AROUSAL_MAP.get(emotion, (0.0, 0.0))
            
            dst_label = os.path.join(data_root, project_split, 'labels', f"{sample_id}.txt")
            with open(dst_label, 'w', encoding='utf-8') as f:
                f.write(f"{emotion}\n")
                f.write(f"{valence},{arousal}\n")
            
            # 生理信号：MELD不包含，保持为空
            
            valid_count += 1
            stats[project_split] += 1
            
            if (valid_count % 500) == 0:
                print(f"  已处理 {valid_count}/{len(samples)} 个样本")
        
        print(f"  {meld_split}集完成：{valid_count} 个有效样本")
    
    # 输出统计信息
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
        text_dir = os.path.join(data_root, split, 'text')
        label_dir = os.path.join(data_root, split, 'labels')
        
        video_count = len([f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]) if os.path.exists(video_dir) else 0
        audio_dir = os.path.join(data_root, split, 'audio')
        meld_audio_count = len([
            f for f in os.listdir(audio_dir)
            if f.startswith('meld_') and f.endswith('.wav') and os.path.isfile(os.path.join(audio_dir, f))
        ]) if os.path.exists(audio_dir) else 0
        text_count = len([f for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f))]) if os.path.exists(text_dir) else 0
        label_count = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]) if os.path.exists(label_dir) else 0
        
        print(f"{split}: 视频={video_count}, meld_wav={meld_audio_count}, 文本={text_count}, 标签={label_count}")

def main():
    """主函数"""
    # 检查MELD数据集是否存在
    if not os.path.exists(MELD_ROOT):
        print(f"错误：MELD数据集目录不存在: {MELD_ROOT}")
        print("请先完成MELD数据集的下载和解压（参考use_data.md 4.5节）")
        return
    
    # 整理数据
    organize_meld_data(MELD_ROOT, DATA_ROOT)

if __name__ == "__main__":
    main()


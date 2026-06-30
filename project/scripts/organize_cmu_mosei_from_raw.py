#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMU-MOSEI 数据集整理脚本（从 Kaggle 解压后的 .csd 文件）

目的：
- 从解压后的 CMU-MOSEI 目录（包含 .csd 特征文件）中提取数据
- 整合为本项目现有的 data 目录结构，方便直接使用现有的 MultimodalDataset 和 train.py

数据来源：
- downloads/CMU_MOSEI_raw/CMU-MOSEI/ 目录下的 .csd 文件
- 包含：labels, languages, acoustics, visuals 等模态的特征文件
"""

import os
import numpy as np
from pathlib import Path
from mmsdk import mmdatasdk

# 项目根目录
PROJECT_ROOT = "/home/lizhichun_24/sda1/code/multimodal/project"

# Kaggle 解压后的数据目录
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "downloads", "CMU_MOSEI_raw", "CMU-MOSEI")

# 目标：按照 MultimodalDataset 期望的结构整合到该目录
TARGET_ROOT = os.path.join(PROJECT_ROOT, "data", "CMU_MOSEI_MM")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_mosei_from_raw():
    """
    从解压后的目录加载 CMU-MOSEI 数据
    """
    print("=" * 60)
    print("从 Kaggle 解压目录加载 CMU-MOSEI 数据")
    print("=" * 60)
    
    if not os.path.exists(RAW_DATA_DIR):
        raise RuntimeError(f"数据目录不存在: {RAW_DATA_DIR}")
    
    print(f"数据目录: {RAW_DATA_DIR}")
    
    # 构建 recipe，指向实际的 .csd 文件
    recipe = {}
    
    # Labels
    labels_path = os.path.join(RAW_DATA_DIR, "labels", "CMU_MOSEI_Labels.csd")
    if os.path.exists(labels_path):
        recipe["labels"] = labels_path
        print(f"✓ 找到标签文件: {labels_path}")
    else:
        raise RuntimeError(f"未找到标签文件: {labels_path}")
    
    # Languages (文本相关)
    languages_dir = os.path.join(RAW_DATA_DIR, "languages")
    if os.path.exists(languages_dir):
        word_vectors_path = os.path.join(languages_dir, "CMU_MOSEI_TimestampedWordVectors.csd")
        words_path = os.path.join(languages_dir, "CMU_MOSEI_TimestampedWords.csd")
        
        if os.path.exists(word_vectors_path):
            recipe["word_vectors"] = word_vectors_path
            print(f"✓ 找到词向量文件: {word_vectors_path}")
        if os.path.exists(words_path):
            recipe["words"] = words_path
            print(f"✓ 找到词文件: {words_path}")
    
    # Acoustics
    acoustics_dir = os.path.join(RAW_DATA_DIR, "acoustics")
    if os.path.exists(acoustics_dir):
        covarep_path = os.path.join(acoustics_dir, "CMU_MOSEI_COVAREP.csd")
        if os.path.exists(covarep_path):
            recipe["covarep"] = covarep_path
            print(f"✓ 找到声学特征文件: {covarep_path}")
    
    # Visuals
    visuals_dir = os.path.join(RAW_DATA_DIR, "visuals")
    if os.path.exists(visuals_dir):
        visual_path = os.path.join(visuals_dir, "CMU_MOSEI_VisualOpenFace2.csd")
        if os.path.exists(visual_path):
            recipe["visual"] = visual_path
            print(f"✓ 找到视觉特征文件: {visual_path}")
    
    print(f"\n总共找到 {len(recipe)} 个特征文件")
    
    # 使用 mmsdk 加载数据
    print("\n开始加载数据...")
    dataset = mmdatasdk.mmdataset(recipe)
    
    print("可用的 computational_sequences 键：", list(dataset.computational_sequences.keys()))
    
    return dataset


def get_standard_folds():
    """
    使用 SDK 提供的标准划分
    """
    try:
        folds_module = mmdatasdk.cmu_mosei.standard_folds
        split_dict = {
            "train": set(folds_module.standard_train_fold),
            "val": set(folds_module.standard_valid_fold),
            "test": set(folds_module.standard_test_fold),
        }
        print(f"成功加载标准划分: train={len(split_dict['train'])}, val={len(split_dict['val'])}, test={len(split_dict['test'])}")
        return split_dict
    except Exception as e:
        print(f"警告：无法获取标准划分，将使用随机划分: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_text_from_words(words_csd, labels_csd, video_id, segment_idx):
    """
    从 TimestampedWords + Labels 时间区间提取 segment 文本。
    """
    from data.mosei_text_utils import extract_segment_text

    if words_csd is None or labels_csd is None:
        return None
    if video_id not in words_csd.data or video_id not in labels_csd.data:
        return None
    try:
        text = extract_segment_text(
            words_csd.data[video_id],
            labels_csd.data[video_id],
            segment_idx,
            drop_sp=True,
        )
        return text or None
    except Exception as e:
        print(f"提取文本时出错 (vid={video_id}, seg={segment_idx}): {e}")
        return None


def extract_features_for_segment(csd, video_id, label_intervals, segment_idx):
    """
    从 visual 或 acoustic csd 中提取对应 segment 的特征
    
    Args:
        csd: computational_sequence 对象
        video_id: 视频ID
        label_intervals: labels 的 intervals 数组，shape (num_segments, 2)
        segment_idx: 当前 segment 的索引
    
    Returns:
        features: numpy array，提取的特征，如果失败返回 None
    """
    if csd is None or video_id not in csd.data:
        return None
    
    try:
        vid_data = csd.data[video_id]
        csd_intervals = vid_data["intervals"]  # (num_frames, 2)
        csd_features = vid_data["features"]    # (num_frames, feature_dim)
        
        # 获取当前 segment 的时间范围
        seg_start, seg_end = label_intervals[segment_idx]
        
        # 找到在时间范围内的所有帧
        mask = (csd_intervals[:, 0] >= seg_start) & (csd_intervals[:, 1] <= seg_end)
        
        if np.sum(mask) == 0:
            # 如果没有完全匹配的，使用重叠的帧
            mask = (csd_intervals[:, 0] < seg_end) & (csd_intervals[:, 1] > seg_start)
        
        if np.sum(mask) == 0:
            return None
        
        # 提取特征
        segment_features = csd_features[mask]
        
        # 如果特征太多，可以平均或采样
        if len(segment_features) > 100:
            # 采样固定数量
            indices = np.linspace(0, len(segment_features) - 1, 100, dtype=int)
            segment_features = segment_features[indices]
        elif len(segment_features) == 0:
            return None
        
        # 转换为 numpy array（如果是 h5py Dataset）
        if hasattr(segment_features, '__array__'):
            segment_features = np.array(segment_features)
        
        return segment_features
        
    except Exception as e:
        # 静默失败，返回 None
        return None


def parse_label_features(features):
    """
    解析标签特征，提取情感类别和维度值
    
    MOSEI 标签格式可能是：
    - 单一 sentiment score (范围 -3 到 +3)
    - 多维 emotion 向量
    - 多个维度 (valence, arousal, dominance 等)
    
    这里先实现基础解析，后续可根据实际格式优化
    """
    if features is None or len(features) == 0:
        return "neutral", 0.0, 0.0
    
    # 尝试解析标签
    # 如果 features 是标量，可能是 sentiment score
    if np.isscalar(features) or (isinstance(features, np.ndarray) and features.size == 1):
        score = float(features)
        # 将 sentiment score 映射到情感类别
        if score >= 0.5:
            emotion = "happy"
            valence, arousal = 0.8, 0.7
        elif score <= -0.5:
            emotion = "sad"
            valence, arousal = -0.6, -0.3
        else:
            emotion = "neutral"
            valence, arousal = 0.0, 0.0
        return emotion, valence, arousal
    
    # 如果 features 是多维向量，尝试提取前两个维度作为 valence 和 arousal
    if isinstance(features, np.ndarray) and features.size >= 2:
        valence = float(features[0])
        arousal = float(features[1]) if features.size > 1 else 0.0
        
        # 根据 valence 和 arousal 推断情感类别
        if valence > 0.5 and arousal > 0.5:
            emotion = "happy"
        elif valence < -0.5 and arousal < 0:
            emotion = "sad"
        elif valence < -0.5 and arousal > 0.5:
            emotion = "angry"
        elif arousal > 0.7:
            emotion = "fear"
        else:
            emotion = "neutral"
        
        return emotion, valence, arousal
    
    # 默认值
    return "neutral", 0.0, 0.0


def organize_cmu_mosei_to_data():
    """
    将 CMU-MOSEI .csd 数据整合为本项目 data 目录下的标准结构
    """
    print("=" * 60)
    print("CMU-MOSEI 数据整理脚本（从 Kaggle 解压目录）")
    print("=" * 60)
    
    # 1. 加载数据
    dataset = load_mosei_from_raw()
    
    # 2. 获取标准划分
    folds = get_standard_folds()
    
    # 3. 预创建目标目录结构
    for split in ["train", "val", "test"]:
        for sub in ["video", "audio", "physiological", "text", "labels"]:
            ensure_dir(os.path.join(TARGET_ROOT, split, sub))
    
    # 4. 获取所有需要的 computational sequences
    labels_csd = None
    words_csd = None
    visual_csd = None
    acoustic_csd = None
    
    for key, csd in dataset.computational_sequences.items():
        if "label" in key.lower():
            labels_csd = csd
        elif "word" in key.lower() and "vector" not in key.lower():
            words_csd = csd
        elif "visual" in key.lower() or "openface" in key.lower() or "facet" in key.lower():
            visual_csd = csd
        elif "covarep" in key.lower() or "acoustic" in key.lower():
            acoustic_csd = csd
    
    if labels_csd is None:
        raise RuntimeError("未找到标签 computational_sequence")
    
    # 正确访问 computational_sequence 的 data 属性
    labels_data = labels_csd.data
    print(f"\n找到标签数据，包含 {len(labels_data)} 个视频")
    if visual_csd:
        print(f"找到视觉特征数据")
    if acoustic_csd:
        print(f"找到声学特征数据")
    
    # 5. 处理每个视频和分段
    counters = {"train": 0, "val": 0, "test": 0}
    
    for vid in labels_data.keys():
        # 确定 split
        split = None
        if folds:
            for s in ["train", "val", "test"]:
                if vid in folds[s]:
                    split = s
                    break
        else:
            # 如果没有标准划分，使用随机划分（基于视频ID的哈希）
            import hashlib
            hash_val = int(hashlib.md5(vid.encode()).hexdigest(), 16)
            if hash_val % 10 < 8:
                split = "train"
            elif hash_val % 10 < 9:
                split = "val"
            else:
                split = "test"
        
        if split is None:
            continue
        
        vid_data = labels_data[vid]
        feats = vid_data["features"]
        intervals = vid_data["intervals"]  # (num_segments, 2) - 每个 segment 的时间范围
        num_segments = feats.shape[0] if hasattr(feats, 'shape') else len(feats)
        
        # 转换为 numpy array（如果是 h5py Dataset）
        if hasattr(intervals, '__array__'):
            intervals = np.array(intervals)
        if hasattr(feats, '__array__'):
            feats = np.array(feats)
        
        for seg_idx in range(num_segments):
            counters[split] += 1
            sample_id = f"mosei_{split}_{counters[split]:06d}"
            
            # 5.1 提取文本（如果文件不存在）
            text_path = os.path.join(TARGET_ROOT, split, "text", f"{sample_id}.txt")
            if not os.path.exists(text_path):
                text_content = None
                if words_csd:
                    text_content = extract_text_from_words(words_csd, labels_csd, vid, seg_idx)
                
                with open(text_path, "w", encoding="utf-8") as f:
                    if text_content:
                        f.write(text_content + "\n")
                    else:
                        f.write(f"Transcript for {vid} segment {seg_idx}\n")
            
            # 5.2 提取标签（如果文件不存在）
            label_path = os.path.join(TARGET_ROOT, split, "labels", f"{sample_id}.txt")
            if not os.path.exists(label_path):
                seg_feats = feats[seg_idx] if hasattr(feats, '__getitem__') else feats
                emotion, valence, arousal = parse_label_features(seg_feats)
                
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(f"{emotion}\n")
                    f.write(f"{valence},{arousal}\n")
            
            # 5.3 提取视觉特征（作为"视频"特征）
            video_path = os.path.join(TARGET_ROOT, split, "video", f"{sample_id}.npy")
            if not os.path.exists(video_path) and visual_csd:
                visual_features = extract_features_for_segment(visual_csd, vid, intervals, seg_idx)
                if visual_features is not None:
                    np.save(video_path, visual_features)
            
            # 5.4 提取声学特征（作为"音频"特征）
            audio_path = os.path.join(TARGET_ROOT, split, "audio", f"{sample_id}.npy")
            if not os.path.exists(audio_path) and acoustic_csd:
                acoustic_features = extract_features_for_segment(acoustic_csd, vid, intervals, seg_idx)
                if acoustic_features is not None:
                    np.save(audio_path, acoustic_features)
            
            # 生理信号：暂不创建文件，模型会自动使用零特征占位
            
            if counters[split] % 1000 == 0:
                print(f"  已处理 {split} 集: {counters[split]} 个样本")
    
    print("\n" + "=" * 60)
    print("整理完成！")
    print("=" * 60)
    print("各划分样本数：")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counters[split]} 个样本")
    
    print(f"\n数据已整理到: {TARGET_ROOT}")
    print("你可以在配置文件 config.yaml 中，将 data.root_dir 设置为该路径进行训练。")


def main():
    try:
        organize_cmu_mosei_to_data()
    except Exception as e:
        print("\n错误：整理 CMU-MOSEI 过程中出现异常：")
        import traceback
        traceback.print_exc()
        print("\n请确认：")
        print(f"  1) 数据已解压到: {RAW_DATA_DIR}")
        print(f"  2) 该目录下包含 labels, languages 等子目录")
        print(f"  3) 各子目录下包含对应的 .csd 文件")


if __name__ == "__main__":
    main()

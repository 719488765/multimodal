"""数据集情感标签映射：unified（跨数据集对齐）与 native（单域原生）。"""

from __future__ import annotations

from typing import Dict, List, Optional

# 跨数据集统一 7 类（happy, sad, angry, fear, neutral, anxious, other）
STANDARD_EMOTION_LABELS: Dict[str, int] = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "fear": 3,
    "neutral": 4,
    "anxious": 5,
    "other": 6,
}

UNIFIED_EMOTION_MAPS: Dict[str, Dict[str, int]] = {
    "crema": {
        "happy": 0,
        "sad": 1,
        "angry": 2,
        "fear": 3,
        "disgust": 6,
        "neutral": 4,
    },
    "meld": {
        "joy": 0,
        "sadness": 1,
        "anger": 2,
        "fear": 3,
        "neutral": 4,
        "surprise": 5,
        "disgust": 6,
    },
    "mosei": {
        "happy": 0,
        "sad": 1,
        "angry": 2,
        "fear": 3,
        "neutral": 4,
        "surprise": 5,
        "disgust": 6,
    },
}

# 单域原生标签空间（不做 surprise→anxious 等跨域映射）
NATIVE_EMOTION_MAPS: Dict[str, Dict[str, int]] = {
    "meld": {
        "joy": 0,
        "sadness": 1,
        "anger": 2,
        "fear": 3,
        "neutral": 4,
        "surprise": 5,
        "disgust": 6,
    },
    "crema": {
        "happy": 0,
        "sad": 1,
        "angry": 2,
        "fear": 3,
        "disgust": 4,
        "neutral": 5,
    },
    "mosei": {
        "happy": 0,
        "sad": 1,
        "angry": 2,
        "fear": 3,
        "neutral": 4,
        "surprise": 5,
        "disgust": 6,
    },
}

NATIVE_EMOTION_NAMES: Dict[str, List[str]] = {
    "meld": ["joy", "sadness", "anger", "fear", "neutral", "surprise", "disgust"],
    "crema": ["happy", "sad", "angry", "fear", "disgust", "neutral"],
    "mosei": ["happy", "sad", "angry", "fear", "neutral", "surprise", "disgust"],
}

UNIFIED_EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]

# native id → unified id（供 Agent 推理映射）
NATIVE_TO_UNIFIED: Dict[str, Dict[int, int]] = {
    "meld": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
    "crema": {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 4},
    "mosei": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
}


def uses_native_labels(dataset_name: Optional[str], datasets_config: dict) -> bool:
    if not dataset_name:
        return False
    ds_cfg = datasets_config.get(dataset_name, {})
    return bool(ds_cfg.get("use_native_labels", False))


def get_sample_emotion_map(
    dataset_name: Optional[str],
    datasets_config: dict,
) -> Dict[str, int]:
    """按样本所属数据集返回标签字符串 → class id 映射。"""
    if not dataset_name:
        return STANDARD_EMOTION_LABELS

    ds_cfg = datasets_config.get(dataset_name, {})
    if ds_cfg.get("use_native_labels", False):
        native = NATIVE_EMOTION_MAPS.get(dataset_name)
        if native:
            return native

    if "emotion_map" in ds_cfg:
        return ds_cfg["emotion_map"]

    if dataset_name in UNIFIED_EMOTION_MAPS:
        return UNIFIED_EMOTION_MAPS[dataset_name]

    return STANDARD_EMOTION_LABELS


def get_emotion_class_names(
    dataset_name: str,
    datasets_config: dict,
    num_classes: Optional[int] = None,
) -> List[str]:
    """返回评估/混淆矩阵用的类别名列表。"""
    if uses_native_labels(dataset_name, datasets_config):
        names = NATIVE_EMOTION_NAMES.get(dataset_name)
        if names:
            return names[: num_classes or len(names)]

    ds_cfg = datasets_config.get(dataset_name, {})
    if "emotion_map" in ds_cfg:
        inv: Dict[int, str] = {}
        for label, idx in ds_cfg["emotion_map"].items():
            inv[idx] = label
        n = num_classes or ds_cfg.get("emotion_classes", max(inv.keys()) + 1 if inv else 7)
        return [inv.get(i, f"class_{i}") for i in range(n)]

    n = num_classes or 7
    return UNIFIED_EMOTION_NAMES[:n]


def native_to_unified(dataset_name: str, native_id: int) -> int:
    """将单域 native 预测 id 映射到 Agent unified 7 类。"""
    mapping = NATIVE_TO_UNIFIED.get(dataset_name, {})
    return mapping.get(native_id, 6)

"""
ASR 文本与多模态情绪 logits 一致性校正（在线部署）。
仅在 ASR 情感词与模型 top 类明显冲突时调整概率，避免盲目覆盖。
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from utils.zh_sentiment_lexicon import (
    LAUGHTER_ONLY_RE,
    intensity_factor,
    match_zh_sentiment,
)

EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]
EMOTION_NAMES_CN = ["开心", "难过", "生气", "害怕", "平静", "焦虑", "其他"]

VALENCE_FLOOR = {
    0: 0.25,
    1: -0.85,
    2: -0.55,
    3: -0.65,
    4: -0.15,
    5: -0.45,
    6: 0.0,
}

FLAT_MAX_PROB_THRESHOLD = 0.38
NON_HAPPY_TOP_IDS = {1, 2, 3, 4, 5, 6}


def match_asr_emotion_target(text: str) -> Optional[int]:
    """从 ASR 文本推断目标情绪 id（供校准与仲裁共用）。"""
    return match_zh_sentiment(text)


_match_target = match_asr_emotion_target


def _is_flat_distribution(probs: List[float], threshold: float = FLAT_MAX_PROB_THRESHOLD) -> bool:
    if not probs:
        return True
    return max(probs) < threshold


def _normalize_probs(probs: List[float]) -> List[float]:
    arr = [max(0.0, float(p)) for p in probs]
    s = sum(arr)
    if s <= 0:
        return [1.0 / len(EMOTION_NAMES)] * len(EMOTION_NAMES)
    return [p / s for p in arr]


def _apply_boost(
    probs: List[float],
    target_id: int,
    boost: float = 0.22,
    *,
    neutral_suppress: float = 0.0,
    suppress_ids: Optional[List[int]] = None,
) -> List[float]:
    n = len(probs)
    if target_id < 0 or target_id >= n:
        return probs
    out = list(probs)
    out[target_id] = out[target_id] + boost
    suppress = suppress_ids if suppress_ids is not None else [4]
    for sid in suppress:
        if sid != target_id and sid < n:
            out[sid] = max(0.0, out[sid] - neutral_suppress)
    return _normalize_probs(out)


def _force_target_emotion(norm: List[float], target_id: int) -> List[float]:
    """ASR 与模型 top 严重冲突时，确保 target 成为 top1。"""
    out = list(norm)
    if target_id < 0 or target_id >= len(out):
        return norm
    out[target_id] = max(out[target_id], 0.52)
    cap_ids = [4, 5, 0, 1, 2, 3, 6]
    for sid in cap_ids:
        if sid != target_id and sid < len(out):
            out[sid] = min(out[sid], 0.22)
    return _normalize_probs(out)


def _force_happy_from_flat(norm: List[float]) -> List[float]:
    return _force_target_emotion(norm, 0)


def _result_from_probs(
    probs: List[float],
    valence: float,
    arousal: float,
) -> Dict[str, Any]:
    emotion_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    valence = max(valence, VALENCE_FLOOR.get(emotion_id, valence))
    if emotion_id == 0:
        valence = max(valence, 0.2)
    return {
        "emotion_id": emotion_id,
        "emotion_label": EMOTION_NAMES[emotion_id],
        "emotion": EMOTION_NAMES[emotion_id],
        "confidence": float(probs[emotion_id]),
        "valence": float(valence),
        "arousal": float(arousal),
        "all_probs": probs,
        "all_probs_labeled": [
            {
                "id": i,
                "label": EMOTION_NAMES[i],
                "label_cn": EMOTION_NAMES_CN[i],
                "prob": float(probs[i]),
            }
            for i in range(len(EMOTION_NAMES))
        ],
    }


def should_calibrate(
    text: str,
    probs: List[float],
    current_id: int,
) -> Tuple[bool, Optional[int], str]:
    target = _match_target(text)
    if target is None:
        return False, None, "no_asr_sentiment_match"

    if len(probs) < len(EMOTION_NAMES):
        probs = probs + [0.0] * (len(EMOTION_NAMES) - len(probs))
    p_target = probs[target] if target < len(probs) else 0.0
    p_neutral = probs[4] if len(probs) > 4 else 0.0
    p_current = probs[current_id] if current_id < len(probs) else 0.0
    flat = _is_flat_distribution(probs)

    if target == current_id and not flat:
        return False, target, "already_aligned"

    # 纯笑声 / 扁平分布 + 正向 ASR
    if target == 0 and flat:
        return True, target, "flat_logits_positive_asr"

    if target == 0 and current_id in NON_HAPPY_TOP_IDS:
        if LAUGHTER_ONLY_RE.match(text.strip()) or (p_neutral - p_target) < 0.55:
            return True, target, "positive_asr_vs_nonhappy_top"

    if target == 0 and current_id == 4:
        if p_target >= 0.03 or (p_neutral - p_target) < 0.62:
            return True, target, "positive_asr_vs_neutral"

    # 负向 ASR：模型 top 为 neutral/anxious/other 时，关键词匹配即可校正
    if target in (1, 2, 3, 5) and current_id in (4, 5, 6):
        return True, target, "negative_asr_vs_neutral_top"

    if target in (1, 2, 3, 5) and current_id in (0, 4, 5, 6):
        if flat or (p_current - p_target) >= 0.15:
            return True, target, "negative_asr_model_conflict"

    if target in (1, 2, 3, 5) and current_id == 4:
        if p_target >= 0.05 or flat:
            return True, target, "negative_asr_vs_neutral"

    if target == 4 and current_id != 4 and flat:
        return True, target, "neutral_asr_vs_flat_wrong_top"

    if p_target >= 0.12 and p_current - p_target < 0.45:
        return True, target, "asr_target_competitive"

    return False, target, "no_conflict"


def apply_asr_emotion_calibration(
    emotion: Dict[str, Any],
    asr_text: str,
    *,
    enabled: bool = True,
) -> Dict[str, Any]:
    if not enabled:
        emotion["asr_calibration_applied"] = False
        emotion["asr_calibration_reason"] = "disabled"
        return emotion

    text = (asr_text or "").strip()
    probs = list(emotion.get("all_probs") or [])
    if not probs:
        emotion["asr_calibration_applied"] = False
        emotion["asr_calibration_reason"] = "empty_probs"
        return emotion

    current_id = int(emotion.get("emotion_id", int(max(range(len(probs)), key=lambda i: probs[i]))))
    ok, target, reason = should_calibrate(text, probs, current_id)
    emotion["asr_calibration_applied"] = False
    emotion["asr_calibration_reason"] = reason
    emotion["calibration_profile"] = reason

    if not ok or target is None:
        return emotion

    emotion["model_probs_before_calibration"] = deepcopy(probs)
    norm = _normalize_probs(probs)

    if reason == "flat_logits_positive_asr" and target == 0:
        boosted = _force_happy_from_flat(norm)
    elif reason in ("positive_asr_vs_neutral", "positive_asr_vs_nonhappy_top") and target == 0:
        boosted = _apply_boost(norm, target, boost=0.55, neutral_suppress=0.12, suppress_ids=[4, 5, 1, 2])
        if int(max(range(len(boosted)), key=lambda i: boosted[i])) != target:
            boosted = _force_happy_from_flat(norm)
    elif reason in (
        "negative_asr_vs_neutral_top",
        "negative_asr_model_conflict",
        "negative_asr_vs_neutral",
        "asr_target_competitive",
    ):
        suppress = [4, 5] if target in (1, 2, 3, 5) else [4]
        boosted = _apply_boost(
            norm,
            target,
            boost=0.48 if reason.startswith("negative") else 0.4,
            neutral_suppress=0.28 if reason.startswith("negative") else 0.2,
            suppress_ids=suppress,
        )
        if reason.startswith("negative") and int(max(range(len(boosted)), key=lambda i: boosted[i])) != target:
            boosted = _force_target_emotion(norm, target)
    elif reason == "neutral_asr_vs_flat_wrong_top":
        boosted = _apply_boost(norm, target, boost=0.35, neutral_suppress=0.15)
    else:
        boosted = _apply_boost(norm, target, boost=0.22 * intensity_factor(text))

    valence = float(emotion.get("valence", 0.0))
    arousal = float(emotion.get("arousal", 0.0))
    updated = _result_from_probs(boosted, valence, arousal)

    prev_label = emotion.get("emotion_label")
    emotion.update(updated)
    emotion["asr_calibration_applied"] = emotion.get("emotion_id") != current_id
    emotion["asr_calibration_reason"] = (
        f"{reason}: {prev_label} -> {emotion.get('emotion_label')}"
        if emotion["asr_calibration_applied"]
        else f"{reason}_no_change"
    )
    emotion["calibration_profile"] = reason
    return emotion

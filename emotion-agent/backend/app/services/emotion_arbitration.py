"""
多模态情绪仲裁：在模型输出、ASR 校准与 ASR 语义之间产生最终展示标签。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]
EMOTION_NAMES_CN = ["开心", "难过", "生气", "害怕", "平静", "焦虑", "其他"]

_POSITIVE_RE = re.compile(
    r"高兴|开心|快乐|愉快|喜悦|兴奋|成功|太好了|真棒|不错|哈哈|呵呵|好开心|很高兴|笑",
    re.I,
)
_NEGATIVE_SAD = re.compile(r"难过|伤心|悲伤|失落|沮丧|想哭|痛苦|不开心|难受|郁闷", re.I)
_NEGATIVE_ANGRY = re.compile(r"生气|愤怒|恼火|气愤|烦死了|讨厌|火大", re.I)
_NEGATIVE_FEAR = re.compile(r"害怕|恐惧|担心|紧张|慌|吓|可怕", re.I)
_NEGATIVE_ANXIOUS = re.compile(r"焦虑|不安|烦躁|压力大|好烦", re.I)
_NEUTRAL_RE = re.compile(r"还好|一般|普通|没什么|平静|冷静|正常", re.I)
_LAUGHTER_ONLY_RE = re.compile(r"^(哈{2,}|呵{2,}|[哈呵]+！*)+$")
_EN_POSITIVE = re.compile(r"\b(happy|glad|joy|great|wonderful|laugh|haha|lol)\b", re.I)
_EN_SAD = re.compile(r"\b(sad|unhappy|depressed|miserable|crying)\b", re.I)
_EN_ANGRY = re.compile(r"\b(angry|mad|furious|annoyed)\b", re.I)
_EN_FEAR = re.compile(r"\b(scared|afraid|fear|worried|nervous)\b", re.I)


def _import_asr_matcher():
    """懒加载 project 侧统一 ASR 情感匹配（与校准模块一致）。"""
    try:
        from utils.asr_emotion_calibration import match_asr_emotion_target

        return match_asr_emotion_target
    except ImportError:
        pass
    for candidate in (
        Path(__file__).resolve().parents[4] / "project",
        Path(__file__).resolve().parents[3] / "project",
    ):
        root = str(candidate.resolve())
        if root not in sys.path and candidate.is_dir():
            sys.path.insert(0, root)
            try:
                from utils.asr_emotion_calibration import match_asr_emotion_target

                return match_asr_emotion_target
            except ImportError:
                continue
    return None


def _asr_sentiment_hint(text: str) -> Optional[int]:
    matcher = _import_asr_matcher()
    if matcher is not None:
        return matcher((text or "").strip())
    return None


def _max_prob(probs: List[float]) -> float:
    return max(probs) if probs else 0.0


def _labeled_probs(probs: List[float]) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "label": EMOTION_NAMES[i],
            "label_cn": EMOTION_NAMES_CN[i],
            "prob": float(probs[i]) if i < len(probs) else 0.0,
        }
        for i in range(len(EMOTION_NAMES))
    ]


def arbitrate_emotion(
    emotion: Dict[str, Any],
    asr_text: str,
    asr_confidence: float = 0.0,
    *,
    flat_threshold: float = 0.38,
    low_conf_threshold: float = 0.42,
    neutral_override_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    写入 emotion 字典：
    - final_emotion_label / final_emotion_id / final_confidence
    - arbitration_source / arbitration_reason
    保留原 model 字段于 model_emotion_*。
    """
    probs = list(emotion.get("all_probs") or [])
    if len(probs) < len(EMOTION_NAMES):
        probs = probs + [0.0] * (len(EMOTION_NAMES) - len(probs))

    model_id = int(emotion.get("model_emotion_id", emotion.get("emotion_id", 4)))
    model_label = emotion.get("model_emotion_label") or emotion.get("emotion_label") or EMOTION_NAMES[model_id]
    model_conf = float(
        emotion.get("model_confidence", emotion.get("confidence") or _max_prob(probs))
    )

    asr_hint = _asr_sentiment_hint(asr_text)
    max_p = _max_prob(probs)
    is_flat = max_p < flat_threshold
    text = (asr_text or "").strip()

    final_id = model_id
    source = "model"
    reason = "model_top1"

    if emotion.get("asr_calibration_applied"):
        final_id = int(emotion.get("emotion_id", model_id))
        source = "asr_calibration"
        reason = str(emotion.get("asr_calibration_reason", "calibrated"))

    # 负向 ASR + 模型 neutral/anxious/other（即使模型高置信也仲裁）
    elif asr_hint in (1, 2, 3, 5) and model_label in ("neutral", "anxious", "other"):
        final_id = asr_hint
        source = "arbitration"
        reason = "negative_asr_over_neutral_model"

    # 正向 ASR + 模型非 happy（扁平或 neutral/sad/anxious）
    elif asr_hint == 0 and (is_flat or model_label in ("neutral", "anxious", "sad", "other", "fear", "angry")):
        if asr_confidence >= 0.35 or _LAUGHTER_ONLY_RE.search(text):
            final_id = 0
            source = "arbitration"
            reason = "positive_asr_over_nonhappy_model"

    # 低置信模型 + 负向 ASR
    elif asr_hint in (1, 2, 3, 5) and model_conf < low_conf_threshold:
        final_id = asr_hint
        source = "arbitration"
        reason = "low_conf_model_with_negative_asr"

    # 模型 happy 但 ASR 明确负向（跨类冲突）
    elif asr_hint in (1, 2, 3) and model_label == "happy" and model_conf < neutral_override_threshold:
        final_id = asr_hint
        source = "arbitration"
        reason = "negative_asr_over_weak_happy"

    # 中性 ASR + 扁平分布 + 模型非 neutral
    elif asr_hint == 4 and is_flat and model_label not in ("neutral",):
        final_id = 4
        source = "arbitration"
        reason = "neutral_asr_over_flat_wrong_top"

    # 极低置信模型 + 高置信 ASR
    elif model_conf < 0.35 and asr_confidence >= 0.65 and asr_hint is not None:
        final_id = asr_hint
        source = "arbitration"
        reason = "high_conf_asr_over_low_conf_model"

    final_label = EMOTION_NAMES[final_id] if final_id < len(EMOTION_NAMES) else "other"
    final_probs = list(probs)
    if final_id != int(max(range(len(final_probs)), key=lambda i: final_probs[i])):
        boost = 0.18
        final_probs[final_id] = final_probs[final_id] + boost
        s = sum(final_probs)
        if s > 0:
            final_probs = [p / s for p in final_probs]

    emotion["final_emotion_id"] = final_id
    emotion["final_emotion_label"] = final_label
    emotion["final_confidence"] = float(final_probs[final_id]) if final_probs else model_conf
    emotion["arbitration_source"] = source
    emotion["arbitration_reason"] = reason
    emotion["model_emotion_id"] = model_id
    emotion["model_emotion_label"] = model_label
    emotion["model_confidence"] = model_conf
    emotion["is_flat_distribution"] = is_flat
    emotion["asr_sentiment_hint"] = EMOTION_NAMES[asr_hint] if asr_hint is not None else None
    emotion["display_probs"] = final_probs
    emotion["display_probs_labeled"] = _labeled_probs(final_probs)

    emotion["emotion_id"] = final_id
    emotion["emotion_label"] = final_label
    emotion["emotion"] = final_label
    emotion["confidence"] = emotion["final_confidence"]
    emotion["all_probs"] = final_probs
    emotion["all_probs_labeled"] = emotion["display_probs_labeled"]

    indexed = sorted(enumerate(final_probs), key=lambda x: -x[1])[:3]
    emotion["top_emotions"] = [
        {"label": EMOTION_NAMES[i], "prob": float(p)} for i, p in indexed if i < len(EMOTION_NAMES)
    ]
    return emotion

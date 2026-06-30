"""ASR 情绪一致性校正单元测试。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.asr_emotion_calibration import apply_asr_emotion_calibration, should_calibrate


def test_positive_asr_vs_neutral_triggers_happy():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.73,
        "valence": -0.36,
        "arousal": 0.2,
        "all_probs": [0.198, 0.01, 0.015, 0.057, 0.73, 0.0, 0.0],
    }
    text = "我很高兴实验取得了成功！哈哈哈哈！"
    ok, target, reason = should_calibrate(text, emotion["all_probs"], 4)
    assert ok is True
    assert target == 0
    out = apply_asr_emotion_calibration(dict(emotion), text)
    assert out["emotion_label"] == "happy"
    assert out["asr_calibration_applied"] is True


def test_flat_logits_laughter_only():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.244,
        "valence": -0.31,
        "arousal": 0.55,
        "all_probs": [0.021, 0.20, 0.158, 0.131, 0.244, 0.204, 0.041],
    }
    text = "哈哈哈哈哈哈"
    ok, target, reason = should_calibrate(text, emotion["all_probs"], 4)
    assert ok is True
    assert reason == "flat_logits_positive_asr"
    out = apply_asr_emotion_calibration(dict(emotion), text)
    assert out["emotion_label"] == "happy"
    assert out["asr_calibration_applied"] is True


def test_flat_top_anxious_with_laughter():
    emotion = {
        "emotion_id": 5,
        "emotion_label": "anxious",
        "confidence": 0.204,
        "valence": -0.2,
        "arousal": 0.5,
        "all_probs": [0.021, 0.20, 0.158, 0.131, 0.244, 0.204, 0.041],
    }
    text = "哈哈哈哈"
    out = apply_asr_emotion_calibration(dict(emotion), text)
    assert out["emotion_label"] == "happy"


def test_no_calibration_neutral_weather():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.8,
        "valence": 0.0,
        "arousal": 0.1,
        "all_probs": [0.05, 0.05, 0.05, 0.05, 0.8, 0.0, 0.0],
    }
    out = apply_asr_emotion_calibration(dict(emotion), "今天阴天，气温十五度。")
    assert out["emotion_label"] == "neutral"
    assert not out.get("asr_calibration_applied")


def test_negative_sad_asr_vs_confident_neutral():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.73,
        "valence": -0.36,
        "arousal": 0.2,
        "all_probs": [0.05, 0.01, 0.015, 0.057, 0.73, 0.0, 0.0],
    }
    text = "我很难过，心里非常难受。"
    ok, target, reason = should_calibrate(text, emotion["all_probs"], 4)
    assert ok is True
    assert target == 1
    out = apply_asr_emotion_calibration(dict(emotion), text)
    assert out["emotion_label"] == "sad"
    assert out["asr_calibration_applied"] is True


def test_negative_angry_asr_vs_neutral():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.62,
        "valence": -0.1,
        "arousal": 0.4,
        "all_probs": [0.04, 0.06, 0.08, 0.05, 0.62, 0.10, 0.05],
    }
    out = apply_asr_emotion_calibration(dict(emotion), "我真的太生气了！")
    assert out["emotion_label"] == "angry"
    assert out["asr_calibration_applied"] is True

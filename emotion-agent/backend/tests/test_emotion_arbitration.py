from app.services.emotion_arbitration import arbitrate_emotion


def test_arbitration_flat_laughter_to_happy():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.244,
        "all_probs": [0.021, 0.20, 0.158, 0.131, 0.244, 0.204, 0.041],
        "asr_calibration_applied": False,
        "model_emotion_label": "neutral",
        "model_emotion_id": 4,
        "model_confidence": 0.244,
    }
    out = arbitrate_emotion(emotion, "哈哈哈哈哈哈", asr_confidence=0.9)
    assert out["final_emotion_label"] == "happy"
    assert out["emotion_label"] == "happy"
    assert out["arbitration_source"] == "arbitration"


def test_arbitration_sad_over_confident_neutral():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.73,
        "all_probs": [0.05, 0.01, 0.015, 0.057, 0.73, 0.0, 0.0],
        "asr_calibration_applied": False,
        "model_emotion_label": "neutral",
        "model_emotion_id": 4,
        "model_confidence": 0.73,
    }
    out = arbitrate_emotion(emotion, "我很难过", asr_confidence=0.85)
    assert out["final_emotion_label"] == "sad"
    assert out["arbitration_source"] == "arbitration"
    assert out["arbitration_reason"] == "negative_asr_over_neutral_model"


def test_arbitration_passthrough_no_asr_sentiment():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.6,
        "all_probs": [0.05, 0.05, 0.05, 0.25, 0.6, 0.0, 0.0],
        "asr_calibration_applied": False,
        "model_emotion_label": "neutral",
        "model_emotion_id": 4,
        "model_confidence": 0.6,
    }
    out = arbitrate_emotion(
        emotion,
        "Are you from over there or download directly",
        asr_confidence=0.8,
    )
    assert out["final_emotion_label"] == "neutral"
    assert out["arbitration_source"] == "model"

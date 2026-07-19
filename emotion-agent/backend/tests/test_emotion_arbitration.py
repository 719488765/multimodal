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


def test_trust_chinese_bert_keeps_model_over_lexicon():
    """中文 BERT AVT：有明确峰值时不得被『我很难过』词典改成 sad。"""
    emotion = {
        "emotion_id": 6,
        "emotion_label": "other",
        "confidence": 0.55,
        "all_probs": [0.05, 0.08, 0.05, 0.05, 0.12, 0.10, 0.55],
        "asr_calibration_applied": False,
        "model_emotion_label": "other",
        "model_emotion_id": 6,
        "model_confidence": 0.55,
    }
    out = arbitrate_emotion(
        emotion,
        "我很难过",
        asr_confidence=0.9,
        flat_threshold=0.32,
        trust_model=True,
    )
    assert out["final_emotion_label"] == "other"
    assert out["arbitration_source"] == "model"
    assert out["arbitration_reason"] == "trust_chinese_bert_avt"


def test_trust_chinese_bert_flat_allows_asr_hint():
    emotion = {
        "emotion_id": 4,
        "emotion_label": "neutral",
        "confidence": 0.22,
        "all_probs": [0.14, 0.15, 0.14, 0.13, 0.22, 0.12, 0.10],
        "asr_calibration_applied": False,
        "model_emotion_label": "neutral",
        "model_emotion_id": 4,
        "model_confidence": 0.22,
    }
    out = arbitrate_emotion(
        emotion,
        "我很难过",
        asr_confidence=0.9,
        flat_threshold=0.32,
        trust_model=True,
    )
    assert out["final_emotion_label"] == "sad"
    assert out["arbitration_reason"] == "flat_logits_asr_hint_chinese_bert"

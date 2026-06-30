from __future__ import annotations

import random
from typing import Any, Dict

from app.adapters.base import EmotionModelAdapter

EMOTIONS = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]


class MockAdapter(EmotionModelAdapter):
    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def infer(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = (sample.get("text") or "").lower()
        if any(k in text for k in ["开心", "高兴", "happy"]):
            label = "happy"
        elif any(k in text for k in ["难过", "sad", "沮丧"]):
            label = "sad"
        elif any(k in text for k in ["生气", "angry"]):
            label = "angry"
        else:
            label = random.choice(EMOTIONS)

        confidence = round(random.uniform(0.45, 0.88), 3)
        probs = [0.05] * len(EMOTIONS)
        probs[EMOTIONS.index(label)] = confidence
        remaining = max(0.0, 1.0 - confidence)
        fill = remaining / (len(EMOTIONS) - 1)
        probs = [fill if i != EMOTIONS.index(label) else confidence for i in range(len(EMOTIONS))]

        return {
            "emotion_label": label,
            "confidence": confidence,
            "valence": round(random.uniform(-1, 1), 3),
            "arousal": round(random.uniform(-1, 1), 3),
            "all_probs": probs,
            "degraded_mode": not bool(sample.get("video_chunk_b64") and sample.get("audio_chunk_b64")),
            "inference_source": "mock_heuristic",
            "checkpoint_preset": "",
            "fusion_strategy": "none",
            "inference_ms": 0.0,
            "emotion_id": EMOTIONS.index(label),
            "checkpoint_file": "",
        }

    def health(self) -> Dict[str, Any]:
        return {"provider": "mock", "loaded": self._loaded}

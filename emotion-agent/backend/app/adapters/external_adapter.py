from __future__ import annotations

from typing import Any, Dict

from app.adapters.base import EmotionModelAdapter


class ExternalAdapter(EmotionModelAdapter):
    """Placeholder for third-party emotion model integration."""

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def infer(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # This is intentionally conservative until external model is wired.
        return {
            "emotion_label": "neutral",
            "confidence": 0.4,
            "valence": 0.0,
            "arousal": 0.0,
            "all_probs": [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1],
            "degraded_mode": True,
        }

    def health(self) -> Dict[str, Any]:
        return {"provider": "external", "loaded": self._loaded, "note": "placeholder"}

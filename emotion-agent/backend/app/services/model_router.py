from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException

from app.adapters.base import EmotionModelAdapter
from app.adapters.current_project_adapter import CurrentProjectAdapter
from app.adapters.external_adapter import ExternalAdapter
from app.adapters.mock_adapter import MockAdapter
from app.core.config import CHECKPOINT_PRESETS, settings

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self) -> None:
        self._provider = settings.model_provider.lower().strip()
        self._default_preset = (settings.model_checkpoint_preset or "sdavt_meld_zh_agent_v2").lower().strip()
        self._adapter_cache: Dict[str, EmotionModelAdapter] = {}
        self._adapter: EmotionModelAdapter = self._build_adapter(self._provider)
        if self._provider == "current":
            self._adapter.load()
            self._adapter_cache[self._default_preset] = self._adapter
            logger.info(
                "[EMOTION_MODEL] startup ok provider=current preset=%s config=%s checkpoint=%s device=%s",
                self._default_preset,
                settings.model_config_path,
                settings.model_checkpoint_path,
                settings.model_device,
            )
        else:
            logger.warning(
                "[EMOTION_MODEL] startup provider=%s (NOT using trained checkpoint; set MODEL_PROVIDER=current)",
                self._provider,
            )

    def _build_adapter(self, provider: str, preset: Optional[str] = None) -> EmotionModelAdapter:
        if provider == "current":
            preset_key = (preset or self._default_preset).lower().strip()
            if preset_key not in CHECKPOINT_PRESETS:
                preset_key = self._default_preset
            paths = CHECKPOINT_PRESETS[preset_key]
            root = Path(settings.project_root).resolve()
            if preset_key == self._default_preset and settings.model_config_path:
                config_path = settings.model_config_path
                checkpoint_path = settings.model_checkpoint_path
            else:
                config_path = str(root / paths["train_config"])
                checkpoint_path = str(root / paths["checkpoint"])
            return CurrentProjectAdapter(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                project_root=str(root),
                device=settings.model_device,
                preset=preset_key,
            )
        if provider == "external":
            return ExternalAdapter()
        return MockAdapter()

    def _resolve_preset(self, sample: dict) -> str:
        meta = sample.get("metadata") or {}
        requested = str(meta.get("checkpoint_preset") or "").lower().strip()
        if requested in CHECKPOINT_PRESETS:
            return requested

        auto = meta.get("auto_preset")
        if auto is True or str(auto).lower() in ("1", "true", "yes"):
            profile = meta.get("inference_profile") or {}
            suggested = str(profile.get("suggested_preset") or "").lower().strip()
            if suggested in CHECKPOINT_PRESETS:
                return suggested
            # fallback: language → zh/en defaults
            lang = str(profile.get("language") or "").lower().strip()
            if lang in ("zh", "mixed"):
                return "sdavt_meld_zh_agent_v2"
            if lang == "en":
                return "sdavt_meld_v3_r4"

        return self._default_preset

    def _get_adapter(self, sample: dict) -> EmotionModelAdapter:
        if self._provider != "current":
            return self._adapter
        preset = self._resolve_preset(sample)
        if preset in self._adapter_cache:
            return self._adapter_cache[preset]
        adapter = self._build_adapter("current", preset=preset)
        adapter.load()
        self._adapter_cache[preset] = adapter
        logger.info("[EMOTION_MODEL] loaded additional preset=%s", preset)
        return adapter

    def preload(self, preset: str) -> dict:
        """Eager-load a preset into GPU cache (for frontend switch warm-up)."""
        preset_key = (preset or "").lower().strip()
        if preset_key not in CHECKPOINT_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")
        if self._provider != "current":
            return {
                "ok": False,
                "preset": preset_key,
                "loaded": False,
                "message": f"provider={self._provider}; preload only for MODEL_PROVIDER=current",
            }
        if preset_key in self._adapter_cache:
            return {
                "ok": True,
                "preset": preset_key,
                "loaded": True,
                "already_cached": True,
                "loaded_presets": list(self._adapter_cache.keys()),
            }
        adapter = self._build_adapter("current", preset=preset_key)
        adapter.load()
        self._adapter_cache[preset_key] = adapter
        logger.info("[EMOTION_MODEL] preload ok preset=%s", preset_key)
        return {
            "ok": True,
            "preset": preset_key,
            "loaded": True,
            "already_cached": False,
            "loaded_presets": list(self._adapter_cache.keys()),
        }

    def infer(self, sample: dict) -> dict:
        adapter = self._get_adapter(sample)
        try:
            result = adapter.infer(sample)
            result["model_provider"] = self._provider
            preset = self._resolve_preset(sample) if self._provider == "current" else ""
            if preset:
                result["checkpoint_preset"] = preset
            if self._provider == "current" and not result.get("inference_source"):
                result["inference_source"] = "checkpoint"
            if self._provider == "mock" and not result.get("inference_source"):
                result["inference_source"] = "mock_heuristic"
            logger.info(
                "[EMOTION_MODEL] infer ok source=%s provider=%s label=%s id=%s conf=%.3f ms=%.1f preset=%s fusion=%s degraded=%s",
                result.get("inference_source"),
                self._provider,
                result.get("emotion_label"),
                result.get("emotion_id"),
                float(result.get("confidence") or 0),
                float(result.get("inference_ms") or 0),
                result.get("checkpoint_preset", ""),
                result.get("fusion_strategy", ""),
                result.get("degraded_mode"),
            )
            return result
        except Exception as exc:
            logger.exception("[EMOTION_MODEL] infer failed provider=%s: %s", self._provider, exc)
            if settings.model_fail_on_error and self._provider == "current":
                raise HTTPException(
                    status_code=503,
                    detail=f"Emotion model inference failed: {exc}",
                ) from exc
            fallback = MockAdapter()
            fallback.load()
            result = fallback.infer(sample)
            result["model_provider"] = "mock"
            result["inference_source"] = "mock_fallback"
            result["degraded_mode"] = True
            logger.warning(
                "[EMOTION_MODEL] infer degraded to mock_fallback after error: %s",
                exc,
            )
            return result

    def health(self) -> dict:
        health = self._adapter.health()
        health["model_provider"] = self._provider
        health["checkpoint_preset"] = self._default_preset
        health["preset"] = self._default_preset
        health["available_presets"] = list(CHECKPOINT_PRESETS.keys())
        health["loaded_presets"] = list(self._adapter_cache.keys())
        return health

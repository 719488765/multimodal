from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from app.adapters.base import EmotionModelAdapter

logger = logging.getLogger(__name__)


class CurrentProjectAdapter(EmotionModelAdapter):
    """加载 project 训练 checkpoint，执行真实多模态推理。"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        project_root: str,
        device: str = "cuda",
        preset: str = "ap2_m1",
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.project_root = project_root
        self.device = device
        self.preset = preset
        self._service = None
        self._load_error: Optional[str] = None

    def _ensure_project_path(self) -> None:
        root = os.path.abspath(self.project_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def load(self) -> None:
        self._ensure_project_path()
        try:
            from utils.emotion_inference_service import EmotionInferenceService

            if self.device == "cpu":
                logger.warning("MODEL_DEVICE=cpu: inference will be slow; GPU recommended for demo.")

            self._service = EmotionInferenceService(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                project_root=self.project_root,
            )
            self._service.load()
            deploy_cfg = Path(self.project_root) / "config" / "config_agent_deploy.yaml"
            if deploy_cfg.is_file():
                with open(deploy_cfg, "r", encoding="utf-8") as f:
                    deploy = yaml.safe_load(f) or {}
                temporal = deploy.get("temporal_inference")
                if temporal:
                    self._service.set_temporal_config(temporal)
                    if self._service.config is not None:
                        self._service.config["temporal_inference"] = temporal
            self._load_error = None
            logger.info(
                "Loaded emotion model preset=%s config=%s checkpoint=%s",
                self.preset,
                self.config_path,
                self.checkpoint_path,
            )
        except Exception as exc:
            self._service = None
            self._load_error = str(exc)
            logger.exception("Failed to load emotion model: %s", exc)
            raise

    def infer(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self._service is None:
            raise RuntimeError(self._load_error or "Emotion model not loaded")
        result = self._service.predict_from_sample(sample)
        result.pop("emotion", None)
        result["checkpoint_preset"] = self.preset
        return result

    def health(self) -> Dict[str, Any]:
        base = {
            "provider": "current",
            "preset": self.preset,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "project_root": self.project_root,
            "load_error": self._load_error,
        }
        if self._service is not None:
            base.update(self._service.health())
        else:
            base["loaded"] = False
            base["config_exists"] = os.path.isfile(self.config_path)
            base["checkpoint_exists"] = os.path.isfile(self.checkpoint_path)
        return base

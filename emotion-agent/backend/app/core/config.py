from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 相对 PROJECT_ROOT 的 preset → (训练 config, checkpoint)
CHECKPOINT_PRESETS: Dict[str, Dict[str, str]] = {
    "ap2_m1": {
        "train_config": "config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml",
        "checkpoint": (
            "checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
    "ap4_w005": {
        "train_config": "config/rerun/accuracy_plan/ap4_config_AVT_DA_w005_accuracy_seq.yaml",
        "checkpoint": (
            "checkpoints_accuracy_seq/AP4_AVT_pretrain_3datasets_DA_w005_20260514_071550/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
    "agent_chinese": {
        "train_config": "config/rerun/accuracy_plan/ap2_M1_chinese_text_agent.yaml",
        "checkpoint": (
            "checkpoints_accuracy_seq/AP2_M1_chinese_text_agent/"
            "checkpoint_finetune_best_f1.pth"
        ),
    },
    "meld_only": {
        "train_config": "config/rerun/accuracy_plan/ap1_AVT_ES_meld_only_s3407.yaml",
        "checkpoint": (
            "checkpoints_accuracy_seq/AP1_AVT_ES_pretrain_meld_only_s3407_20260420_202232/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
    "mosei_only": {
        "train_config": "config/rerun/accuracy_plan/ap1_AVT_ES_mosei_only_s3407.yaml",
        "checkpoint": (
            "checkpoints_accuracy_seq/AP1_AVT_ES_pretrain_mosei_only_s3407_20260420_203623/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
    "sdavt_meld_v3_r4": {
        "train_config": "config/sdavt_v3_r4/p3_m3/meld/M3_M3_uniform.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M3_uniform_20260626_031222/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
}


def _default_project_root() -> str:
    backend_dir = Path(__file__).resolve().parents[2]
    candidate = backend_dir.parent.parent / "project"
    if candidate.is_dir():
        return str(candidate)
    return str(backend_dir.parent / "project")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Emotion Agent Backend"
    app_env: str = "dev"
    model_provider: str = "mock"
    model_checkpoint_preset: str = "meld_only"
    project_root: str = ""
    model_device: str = "cuda"
    llm_provider: str = "template"
    llm_model: str = ""
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_timeout_sec: float = 20.0
    llm_temperature: float = 0.6
    asr_provider: str = "whisper_api"
    asr_whisper_api_url: str = "http://127.0.0.1:9010/v1/audio/transcriptions"
    asr_whisper_api_key: str = ""
    asr_whisper_api_model: str = "small"
    asr_whisper_api_language: str = "zh"
    asr_whisper_local_model: str = "base"
    asr_timeout_sec: float = 45.0
    asr_max_retries: int = 1
    model_config_path: str = ""
    model_checkpoint_path: str = ""
    server_base_url: str = "http://localhost:8000"
    cors_allow_origin: str = "*"
    model_fail_on_error: bool = True

    @model_validator(mode="after")
    def resolve_model_paths(self) -> "Settings":
        root = Path(self.project_root or _default_project_root()).resolve()
        self.project_root = str(root)

        preset = (self.model_checkpoint_preset or "ap2_m1").lower().strip()
        if preset not in CHECKPOINT_PRESETS:
            preset = "ap2_m1"
            self.model_checkpoint_preset = preset

        preset_paths = CHECKPOINT_PRESETS[preset]
        if not self.model_config_path:
            self.model_config_path = str(root / preset_paths["train_config"])
        elif not os.path.isabs(self.model_config_path):
            self.model_config_path = str((root / self.model_config_path).resolve())

        if not self.model_checkpoint_path:
            self.model_checkpoint_path = str(root / preset_paths["checkpoint"])
        elif not os.path.isabs(self.model_checkpoint_path):
            self.model_checkpoint_path = str((root / self.model_checkpoint_path).resolve())

        return self


settings = Settings()

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
        "train_config": "config/sdavt_v3_r4/p3_m3/meld/M3_M7_combo.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
    "sdavt_meld_zh_agent": {
        "train_config": "config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/"
            "checkpoint_finetune_best_f1.pth"
        ),
    },
    "sdavt_mosei_r4": {
        "train_config": "config/sdavt_v3_r4/p2_fusion/mosei/F_O_ES_emotion_shift.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_F_O_ES_20260624_101647/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
    "sdavt_crema_r4": {
        "train_config": "config/sdavt_v3_r4/p3_c3/crema/C3_C2_w2v_large.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_C3_C2_w2v_large_20260626_004150/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
}


# 前端 /model/status 展示用摘要（实验对照 preset 标注 experimental）
PRESET_METADATA: Dict[str, Dict[str, Any]] = {
    "sdavt_meld_v3_r4": {
        "label": "sdavt_meld_v3_r4（R4 冠军 M3_M7 F1=0.696，推荐）",
        "dataset": "meld",
        "best_f1": 0.696,
        "best_acc": 0.712,
        "recommended": True,
        "experimental": False,
    },
    "sdavt_meld_zh_agent": {
        "label": "sdavt_meld_zh_agent（M3_M7 + 中文 BERT，中文场景推荐）",
        "dataset": "meld",
        "recommended": True,
        "experimental": False,
    },
    "meld_only": {
        "label": "meld_only（MELD 单域 AP1，对照）",
        "dataset": "meld",
        "best_f1": 0.54,
        "recommended": False,
        "experimental": False,
    },
    "ap2_m1": {
        "label": "ap2_m1（三混合 F1≈0.56）",
        "dataset": "mixed",
        "best_f1": 0.56,
        "recommended": False,
        "experimental": False,
    },
    "agent_chinese": {
        "label": "agent_chinese（中文 BERT 微调）",
        "dataset": "mixed",
        "recommended": False,
        "experimental": False,
    },
    "mosei_only": {
        "label": "mosei_only（MOSEI 单域 AP1，实验）",
        "dataset": "mosei",
        "recommended": False,
        "experimental": True,
    },
    "ap4_w005": {
        "label": "ap4_w005（DA 预训练）",
        "dataset": "mixed",
        "recommended": False,
        "experimental": False,
    },
    "sdavt_mosei_r4": {
        "label": "sdavt_mosei_r4（R4 MOSEI F_O_ES F1=0.679，仅实验）",
        "dataset": "mosei",
        "best_f1": 0.679,
        "recommended": False,
        "experimental": True,
    },
    "sdavt_crema_r4": {
        "label": "sdavt_crema_r4（R4 CREMA C3_C2 Acc=0.567，仅实验）",
        "dataset": "crema",
        "best_acc": 0.567,
        "recommended": False,
        "experimental": True,
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

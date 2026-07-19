from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

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
    "sdavt_meld_zh_agent_v2": {
        "train_config": "config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent_v2.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/"
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
        "train_config": "config/sdavt_v3_r4/p3_c_plus/crema/C4_C3_c3_warmstart_acc.yaml",
        "checkpoint": (
            "checkpoints_sdavt_v3_r4/SDAVT_R4_C4_C3_c3_warmstart_acc/"
            "checkpoint_pretrain_best_f1.pth"
        ),
    },
}

# 中文 BERT 骨干 preset：在线必须保留文本模态（AVT）
CHINESE_BERT_PRESETS = frozenset({"sdavt_meld_zh_agent", "sdavt_meld_zh_agent_v2"})

# 英文 Transformer 骨干：中文 ASR 时可选 skip_text
ENGLISH_BACKBONE_PRESETS = frozenset(
    {
        "sdavt_meld_v3_r4",
        "sdavt_mosei_r4",
        "sdavt_crema_r4",
        "meld_only",
        "mosei_only",
        "ap2_m1",
        "ap4_w005",
    }
)

# 前端 /model/status 展示用摘要（AVT 优先级列表）
PRESET_METADATA: Dict[str, Dict[str, Any]] = {
    "sdavt_meld_zh_agent_v2": {
        "label": "中文 Agent v2（AVT）",
        "dataset": "meld",
        "language": "zh",
        "group": "recommended",
        "group_label": "推荐部署",
        "modalities": "AVT",
        "priority": 0,
        "best_f1": 0.6114,
        "best_acc": 0.6363,
        "recommended": True,
        "experimental": False,
        "ui_visible": True,
        "text_backbone": "bert-base-chinese",
    },
    "sdavt_meld_v3_r4": {
        "label": "英文 MELD 冠军 M3_M7（AVT）",
        "dataset": "meld",
        "language": "en",
        "group": "recommended",
        "group_label": "推荐部署",
        "modalities": "AVT",
        "priority": 1,
        "best_f1": 0.6957,
        "best_acc": 0.7121,
        "recommended": True,
        "experimental": False,
        "ui_visible": True,
        "text_backbone": "roberta-base",
    },
    "sdavt_meld_zh_agent": {
        "label": "中文 Agent v1（AVT）",
        "dataset": "meld",
        "language": "zh",
        "group": "chinese",
        "group_label": "中文对照",
        "modalities": "AVT",
        "priority": 2,
        "best_f1": 0.6010,
        "best_acc": 0.6273,
        "recommended": False,
        "experimental": False,
        "ui_visible": True,
        "text_backbone": "bert-base-chinese",
    },
    "sdavt_mosei_r4": {
        "label": "MOSEI F_O_ES（AVT）",
        "dataset": "mosei",
        "language": "en",
        "group": "experimental",
        "group_label": "实验",
        "modalities": "AVT",
        "priority": 3,
        "best_f1": 0.6792,
        "best_acc": 0.7269,
        "recommended": False,
        "experimental": True,
        "ui_visible": True,
        "text_backbone": "bert-base-uncased",
    },
    "sdavt_crema_r4": {
        "label": "CREMA Warmstart C4_C3（AVT）",
        "dataset": "crema",
        "language": "en",
        "group": "experimental",
        "group_label": "实验",
        "modalities": "AVT",
        "priority": 4,
        "best_f1": 0.6057,
        "best_acc": 0.6048,
        "recommended": False,
        "experimental": True,
        "ui_visible": True,
        "text_backbone": "bert-base-uncased",
    },
    "ap2_m1": {
        "label": "三混合 AP2-M1（AVT）",
        "dataset": "mixed",
        "language": "en",
        "group": "legacy",
        "group_label": "历史",
        "modalities": "AVT",
        "priority": 5,
        "best_f1": 0.56,
        "best_acc": 0.61,
        "recommended": False,
        "experimental": False,
        "ui_visible": True,
        "text_backbone": "bert-base-uncased",
    },
    "meld_only": {
        "label": "meld_only（MELD 单域 AP1）",
        "dataset": "meld",
        "language": "en",
        "group": "hidden",
        "group_label": "高级",
        "modalities": "AVT",
        "priority": 90,
        "best_f1": 0.54,
        "recommended": False,
        "experimental": False,
        "ui_visible": False,
    },
    "mosei_only": {
        "label": "mosei_only（MOSEI 单域 AP1）",
        "dataset": "mosei",
        "language": "en",
        "group": "hidden",
        "group_label": "高级",
        "modalities": "AVT",
        "priority": 91,
        "recommended": False,
        "experimental": True,
        "ui_visible": False,
    },
    "agent_chinese": {
        "label": "agent_chinese（遗留 AP2 中文）",
        "dataset": "mixed",
        "language": "zh",
        "group": "hidden",
        "group_label": "高级",
        "modalities": "AVT",
        "priority": 92,
        "recommended": False,
        "experimental": False,
        "ui_visible": False,
    },
    "ap4_w005": {
        "label": "ap4_w005（DA 预训练）",
        "dataset": "mixed",
        "language": "en",
        "group": "hidden",
        "group_label": "高级",
        "modalities": "AVT",
        "priority": 93,
        "best_f1": 0.528,
        "recommended": False,
        "experimental": False,
        "ui_visible": False,
    },
}


def list_available_presets(*, include_hidden: bool = False) -> List[Dict[str, Any]]:
    """按 priority 排序的 preset 列表，供 /model/status 与前端下拉使用。"""
    items: List[Dict[str, Any]] = []
    for preset_id, paths in CHECKPOINT_PRESETS.items():
        meta = PRESET_METADATA.get(preset_id, {})
        ui_visible = bool(meta.get("ui_visible", False))
        if not include_hidden and not ui_visible:
            continue
        priority = int(meta.get("priority", 99))
        best_f1 = meta.get("best_f1")
        best_acc = meta.get("best_acc")
        label = meta.get("label", preset_id)
        metric_parts = []
        if best_f1 is not None:
            metric_parts.append(f"F1={float(best_f1):.3f}")
        if best_acc is not None:
            metric_parts.append(f"Acc={float(best_acc):.3f}")
        metric_str = " / ".join(metric_parts) if metric_parts else ""
        display = f"[P{priority}] {label}"
        if metric_str:
            display = f"{display}｜{metric_str}"
        if meta.get("recommended"):
            display = f"{display} [推荐]"
        elif meta.get("experimental"):
            display = f"{display} [实验]"
        items.append(
            {
                "id": preset_id,
                "train_config": paths.get("train_config"),
                "checkpoint": paths.get("checkpoint"),
                "label": label,
                "display_label": display,
                "best_f1": best_f1,
                "best_acc": best_acc,
                "recommended": bool(meta.get("recommended")),
                "experimental": bool(meta.get("experimental")),
                "priority": priority,
                "group": meta.get("group", "other"),
                "group_label": meta.get("group_label", "其他"),
                "language": meta.get("language"),
                "modalities": meta.get("modalities", "AVT"),
                "ui_visible": ui_visible,
                "text_backbone": meta.get("text_backbone"),
            }
        )
    items.sort(key=lambda x: (x["priority"], x["id"]))
    return items


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
    model_checkpoint_preset: str = "sdavt_meld_zh_agent_v2"
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

        preset = (self.model_checkpoint_preset or "sdavt_meld_zh_agent_v2").lower().strip()
        if preset not in CHECKPOINT_PRESETS:
            preset = "sdavt_meld_zh_agent_v2"
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

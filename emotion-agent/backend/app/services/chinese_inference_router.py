"""
中文感知推理路由：语言检测 + 模态权重 + text bypass / leader_audio + 语言→preset 建议。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import (
    CHINESE_BERT_PRESETS,
    ENGLISH_BACKBONE_PRESETS,
)

PRESET_ZH = "sdavt_meld_zh_agent_v2"
PRESET_EN = "sdavt_meld_v3_r4"


def _import_detect_language():
    try:
        from utils.zh_sentiment_lexicon import detect_language

        return detect_language
    except ImportError:
        pass
    for candidate in (
        Path(__file__).resolve().parents[4] / "project",
        Path(__file__).resolve().parents[3] / "project",
    ):
        root = str(candidate.resolve())
        if root not in sys.path and candidate.is_dir():
            sys.path.insert(0, root)
            try:
                from utils.zh_sentiment_lexicon import detect_language

                return detect_language
            except ImportError:
                continue

    def _fallback(text: str, default: str = "zh") -> str:
        if not (text or "").strip():
            return "unknown"
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[a-zA-Z]{3,}", text):
            return "en"
        return default

    return _fallback


def suggest_preset_for_language(language: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    按语言建议 checkpoint preset。
    若 metadata 已显式指定 checkpoint_preset，返回空串（manual 模式）。
    """
    meta = metadata or {}
    explicit = str(meta.get("checkpoint_preset") or "").lower().strip()
    if explicit:
        return ""
    lang = (language or "").lower().strip()
    if lang in ("zh", "mixed"):
        return PRESET_ZH
    if lang == "en":
        return PRESET_EN
    return PRESET_ZH


def build_inference_profile(
    asr_text: str,
    user_text: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    deploy_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    返回写入 sample.metadata.inference_profile 的路由决策。
    """
    meta = metadata or {}
    deploy = deploy_cfg or {}
    zh_cfg = deploy.get("chinese_agent") or {}
    if meta.get("chinese_agent") is False or zh_cfg.get("enabled") is False:
        return {"enabled": False, "language": "unknown"}

    detect_language = _import_detect_language()
    merged = (user_text or "").strip() or (asr_text or "").strip()
    language = meta.get("language") or detect_language(
        merged, default=str(zh_cfg.get("default_language", "zh"))
    )
    text_source = meta.get("text_source") or (
        "user_input" if (user_text or "").strip() else ("asr" if (asr_text or "").strip() else "empty")
    )

    suggested = suggest_preset_for_language(str(language), meta)
    explicit_preset = str(meta.get("checkpoint_preset") or "").lower().strip()
    # 决定「将要/正在」使用的 preset（用于 skip_text 策略）
    active_preset = explicit_preset or suggested or PRESET_ZH
    preset_mode = "manual" if explicit_preset else "auto"

    # 中文 BERT preset：强制保留文本（AVT）；仅英文骨干 + 中文 ASR 时可 skip
    skip_cfg = zh_cfg.get("skip_text_on_zh_asr_english_backbone_only")
    if skip_cfg is None:
        # 旧键 skip_text_on_zh_asr=true 视为「仅对英文骨干」；false 则永不 skip
        skip_cfg = bool(zh_cfg.get("skip_text_on_zh_asr", False))
    allow_skip_english = bool(skip_cfg)

    skip_text = False
    if (
        allow_skip_english
        and language in ("zh", "mixed")
        and text_source == "asr"
        and not meta.get("force_text_modality")
        and active_preset in ENGLISH_BACKBONE_PRESETS
        and active_preset not in CHINESE_BERT_PRESETS
    ):
        skip_text = True

    # 中文 BERT 绝对禁止 skip
    if active_preset in CHINESE_BERT_PRESETS:
        skip_text = False

    leader = None
    if language in ("zh", "mixed"):
        leader = str(zh_cfg.get("leader_modal_zh", "audio"))
        if leader.lower() == "none":
            leader = None

    # 中文 BERT：提高文本权重；英文骨干遇中文：低权重
    if active_preset in CHINESE_BERT_PRESETS:
        weight_zh = float(zh_cfg.get("text_modality_weight_zh_bert", zh_cfg.get("text_modality_weight_zh", 0.85)))
    else:
        weight_zh = float(zh_cfg.get("text_modality_weight_zh", 0.15))
    text_weight = 1.0 if language == "en" else weight_zh

    profile = {
        "enabled": True,
        "language": language,
        "text_source": text_source,
        "text_modality_weight": text_weight,
        "skip_text_encoder": skip_text,
        "leader_override": leader,
        "flat_threshold": float(zh_cfg.get("flat_threshold", 0.38)),
        "suggested_preset": suggested,
        "preset_mode": preset_mode,
        "active_preset_hint": active_preset,
    }
    return profile

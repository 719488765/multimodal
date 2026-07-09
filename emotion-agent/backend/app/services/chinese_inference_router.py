"""
中文感知推理路由：语言检测 + 模态权重 + text bypass / leader_audio。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional


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

    skip_on_zh = bool(zh_cfg.get("skip_text_on_zh_asr", True))
    skip_text = (
        language in ("zh", "mixed")
        and skip_on_zh
        and text_source == "asr"
        and not meta.get("force_text_modality")
    )

    leader = None
    if language in ("zh", "mixed"):
        leader = str(zh_cfg.get("leader_modal_zh", "audio"))
        if leader.lower() == "none":
            leader = None

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
    }
    return profile

"""Tests for language-based preset suggestion."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.chinese_inference_router import (  # noqa: E402
    PRESET_EN,
    PRESET_ZH,
    build_inference_profile,
    suggest_preset_for_language,
)


def test_suggest_preset_zh():
    assert suggest_preset_for_language("zh", {}) == PRESET_ZH
    assert suggest_preset_for_language("mixed", {}) == PRESET_ZH


def test_suggest_preset_en():
    assert suggest_preset_for_language("en", {}) == PRESET_EN


def test_suggest_preset_manual_override():
    assert suggest_preset_for_language("zh", {"checkpoint_preset": "sdavt_meld_v3_r4"}) == ""


def test_build_inference_profile_includes_suggested_preset():
    profile = build_inference_profile(
        asr_text="我很难过",
        user_text="",
        metadata={"text_source": "asr"},
        deploy_cfg={"chinese_agent": {"enabled": True}},
    )
    assert profile["language"] in ("zh", "mixed")
    assert profile["suggested_preset"] == PRESET_ZH
    assert profile["preset_mode"] == "auto"


def test_build_inference_profile_manual_mode():
    profile = build_inference_profile(
        asr_text="I am sad",
        user_text="",
        metadata={"checkpoint_preset": "sdavt_meld_v3_r4"},
        deploy_cfg={"chinese_agent": {"enabled": True}},
    )
    assert profile["suggested_preset"] == ""
    assert profile["preset_mode"] == "manual"


def test_zh_bert_preset_keeps_text_modality():
    profile = build_inference_profile(
        asr_text="我很难过",
        user_text="",
        metadata={"text_source": "asr", "checkpoint_preset": "sdavt_meld_zh_agent_v2"},
        deploy_cfg={
            "chinese_agent": {
                "enabled": True,
                "skip_text_on_zh_asr_english_backbone_only": True,
            }
        },
    )
    assert profile["skip_text_encoder"] is False
    assert profile["text_modality_weight"] >= 0.5


def test_english_backbone_may_skip_text_on_zh_asr():
    profile = build_inference_profile(
        asr_text="我很难过",
        user_text="",
        metadata={"text_source": "asr", "checkpoint_preset": "sdavt_meld_v3_r4"},
        deploy_cfg={
            "chinese_agent": {
                "enabled": True,
                "skip_text_on_zh_asr_english_backbone_only": True,
            }
        },
    )
    assert profile["language"] in ("zh", "mixed")
    assert profile["skip_text_encoder"] is True

"""Tests for auto preset resolution in ModelRouter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.model_router import ModelRouter  # noqa: E402


def _router_with_default(default: str = "sdavt_meld_zh_agent") -> ModelRouter:
    with patch("app.services.model_router.settings") as mock_settings:
        mock_settings.model_provider = "current"
        mock_settings.model_checkpoint_preset = default
        mock_settings.model_config_path = ""
        mock_settings.model_checkpoint_path = ""
        mock_settings.model_device = "cpu"
        mock_settings.project_root = str(BACKEND_ROOT.parent.parent / "project")
        mock_settings.model_fail_on_error = False
        router = ModelRouter.__new__(ModelRouter)
        router._provider = "current"
        router._default_preset = default
        router._adapter_cache = {}
        router._adapter = MagicMock()
        return router


def test_resolve_explicit_preset():
    router = _router_with_default()
    preset = router._resolve_preset(
        {"metadata": {"checkpoint_preset": "sdavt_meld_v3_r4"}}
    )
    assert preset == "sdavt_meld_v3_r4"


def test_resolve_auto_preset_zh():
    router = _router_with_default()
    preset = router._resolve_preset(
        {
            "metadata": {
                "auto_preset": True,
                "inference_profile": {
                    "language": "zh",
                    "suggested_preset": "sdavt_meld_zh_agent_v2",
                },
            }
        }
    )
    assert preset == "sdavt_meld_zh_agent_v2"


def test_resolve_auto_preset_en():
    router = _router_with_default()
    preset = router._resolve_preset(
        {
            "metadata": {
                "auto_preset": True,
                "inference_profile": {
                    "language": "en",
                    "suggested_preset": "sdavt_meld_v3_r4",
                },
            }
        }
    )
    assert preset == "sdavt_meld_v3_r4"

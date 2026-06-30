"""Unit tests for temporal window inference utilities."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.temporal_inference import (  # noqa: E402
    aggregate_window_predictions,
    audio_duration_sec,
    compute_windows,
    recency_weights,
    should_use_temporal,
    slice_audio_window_tensor,
)


def test_should_use_temporal_short_clip():
    assert should_use_temporal(3.0, window_sec=3.0, margin_sec=0.2) is False
    assert should_use_temporal(3.5, window_sec=3.0, margin_sec=0.2) is True


def test_compute_windows_30s():
    windows = compute_windows(30.0, window_sec=3.0, stride_sec=3.0, max_windows=10)
    assert len(windows) == 10
    assert windows[0].start_sec == 0.0
    assert windows[-1].start_sec == 27.0
    assert windows[-1].end_sec == 30.0


def test_compute_windows_partial_tail():
    windows = compute_windows(7.0, window_sec=3.0, stride_sec=3.0, max_windows=10)
    assert len(windows) == 3
    assert windows[-1].end_sec == 7.0


def test_recency_weights_favor_later():
    w = recency_weights(4, alpha=1.5)
    assert w[-1] > w[0]
    assert abs(w.sum() - 1.0) < 1e-6


def test_aggregate_recency_weighted():
    windows = [
        {"all_probs": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], "confidence": 0.9, "valence": -0.5, "arousal": 0.2},
        {"all_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "confidence": 1.0, "valence": 0.5, "arousal": 0.8},
    ]
    out = aggregate_window_predictions(windows, strategy="recency_weighted", recency_alpha=2.0)
    assert out["emotion_label"] == "other"
    assert out["confidence"] > 0.5


def test_slice_audio_window_tensor_pad():
    sr = 16000
    wave = np.ones(sr * 2, dtype=np.float32) * 0.1
    t = slice_audio_window_tensor(wave, sr, offset_sec=0.0, duration_sec=3.0)
    assert t.shape == (sr * 3,)
    short = np.ones(sr, dtype=np.float32)
    t2 = slice_audio_window_tensor(short, sr, offset_sec=0.0, duration_sec=3.0)
    assert t2.shape == (sr * 3,)


def test_audio_duration_sec():
    wave = np.zeros(16000 * 5, dtype=np.float32)
    assert abs(audio_duration_sec(wave, 16000) - 5.0) < 0.01

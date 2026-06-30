"""
长时多窗口情绪推理：将完整采集切分为多个训练一致的 3s 窗口，再聚合概率。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None

try:
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover
    sf = None

EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]
EMOTION_NAMES_CN = ["开心", "难过", "生气", "害怕", "平静", "焦虑", "其他"]

DEFAULT_TEMPORAL_CFG: Dict[str, Any] = {
    "enabled": True,
    "window_sec": 3.0,
    "stride_sec": 3.0,
    "max_windows": 10,
    "max_capture_sec": 30.0,
    "max_capture_sec_cloudflare": 12.0,
    "aggregation": "recency_weighted",
    "recency_alpha": 1.5,
    "batch_windows": True,
    "short_path_margin_sec": 0.2,
}


@dataclass(frozen=True)
class WindowSpec:
    index: int
    start_sec: float
    end_sec: float


def merge_temporal_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_TEMPORAL_CFG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def should_use_temporal(
    total_sec: float,
    window_sec: float = 3.0,
    margin_sec: float = 0.2,
    enabled: bool = True,
) -> bool:
    if not enabled:
        return False
    return total_sec > window_sec + margin_sec


def compute_windows(
    total_sec: float,
    window_sec: float = 3.0,
    stride_sec: float = 3.0,
    max_windows: int = 10,
) -> List[WindowSpec]:
    if total_sec <= 0:
        return [WindowSpec(0, 0.0, window_sec)]

    stride_sec = max(stride_sec, 0.1)
    window_sec = max(window_sec, 0.1)
    windows: List[WindowSpec] = []
    offset = 0.0
    idx = 0
    while offset < total_sec - 1e-6 and len(windows) < max_windows:
        start = offset
        end = min(offset + window_sec, total_sec)
        windows.append(WindowSpec(index=idx, start_sec=start, end_sec=end))
        idx += 1
        if end >= total_sec:
            break
        offset += stride_sec

    if not windows:
        windows.append(WindowSpec(0, 0.0, min(window_sec, total_sec)))
    return windows


def load_audio_waveform(
    audio_bytes: bytes,
    sample_rate: int = 16000,
) -> Optional[np.ndarray]:
    if not audio_bytes:
        return None

    audio = None
    if sf is not None:
        try:
            audio, file_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != sample_rate and librosa is not None:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sample_rate)
            elif file_sr != sample_rate:
                ratio = sample_rate / float(file_sr)
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
        except Exception:
            audio = None

    if audio is None and librosa is not None:
        try:
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate)
        except Exception:
            return None

    if audio is None:
        return None
    return np.asarray(audio, dtype=np.float32)


def audio_duration_sec(waveform: np.ndarray, sample_rate: int) -> float:
    if waveform is None or len(waveform) == 0:
        return 0.0
    return len(waveform) / float(sample_rate)


def slice_audio_window_tensor(
    waveform: np.ndarray,
    sample_rate: int,
    offset_sec: float,
    duration_sec: float,
) -> torch.Tensor:
    target_length = int(sample_rate * duration_sec)
    start = int(offset_sec * sample_rate)
    end = start + target_length
    chunk = waveform[start:end]
    if len(chunk) < target_length:
        chunk = np.pad(chunk, (0, target_length - len(chunk)), mode="constant")
    elif len(chunk) > target_length:
        chunk = chunk[:target_length]
    return torch.from_numpy(chunk).float()


def recency_weights(n: int, alpha: float = 1.5) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    weights = np.array([(i + 1) ** alpha for i in range(n)], dtype=np.float64)
    return weights / weights.sum()


def aggregate_window_predictions(
    window_results: Sequence[Dict[str, Any]],
    strategy: str = "recency_weighted",
    recency_alpha: float = 1.5,
) -> Dict[str, Any]:
    if not window_results:
        return {
            "emotion_id": 4,
            "emotion_label": "neutral",
            "confidence": 0.0,
            "valence": 0.0,
            "arousal": 0.0,
            "all_probs": [0.0] * len(EMOTION_NAMES),
        }

    probs_list = [np.asarray(w.get("all_probs") or [], dtype=np.float64) for w in window_results]
    num_classes = max(len(p) for p in probs_list) if probs_list else len(EMOTION_NAMES)
    probs_list = [
        p if len(p) == num_classes else np.pad(p, (0, num_classes - len(p)))
        for p in probs_list
    ]
    valences = [float(w.get("valence", 0.0)) for w in window_results]
    arousals = [float(w.get("arousal", 0.0)) for w in window_results]

    n = len(window_results)
    if strategy == "mean":
        weights = np.ones(n, dtype=np.float64) / n
    elif strategy == "peak":
        confidences = [float(w.get("confidence", 0.0)) for w in window_results]
        best = int(np.argmax(confidences))
        return dict(window_results[best])
    else:
        weights = recency_weights(n, alpha=recency_alpha)

    stacked = np.stack(probs_list, axis=0)
    final_probs = (stacked * weights[:, None]).sum(axis=0)
    final_probs = np.clip(final_probs, 0.0, 1.0)
    if final_probs.sum() > 0:
        final_probs = final_probs / final_probs.sum()

    emotion_id = int(np.argmax(final_probs))
    return {
        "emotion_id": emotion_id,
        "emotion_label": EMOTION_NAMES[emotion_id] if emotion_id < len(EMOTION_NAMES) else "other",
        "emotion": EMOTION_NAMES[emotion_id] if emotion_id < len(EMOTION_NAMES) else "other",
        "confidence": float(final_probs[emotion_id]),
        "valence": float(np.dot(weights, valences)),
        "arousal": float(np.dot(weights, arousals)),
        "all_probs": final_probs.tolist(),
    }


def build_temporal_summary(
    window_results: Sequence[Dict[str, Any]],
    windows: Sequence[WindowSpec],
    aggregation: str,
    total_sec: float,
) -> Dict[str, Any]:
    labels = [w.get("emotion_label", "neutral") for w in window_results]
    first_label = labels[0] if labels else "neutral"
    last_label = labels[-1] if labels else "neutral"
    dominant = max(set(labels), key=labels.count) if labels else "neutral"
    shift = first_label != last_label
    return {
        "mode": "temporal",
        "aggregation": aggregation,
        "total_duration_sec": round(total_sec, 2),
        "num_windows": len(window_results),
        "window_sec": windows[0].end_sec - windows[0].start_sec if windows else 3.0,
        "dominant_window_emotion": dominant,
        "first_window_emotion": first_label,
        "last_window_emotion": last_label,
        "emotion_shift_detected": shift,
    }


def format_window_results(
    raw_results: Sequence[Dict[str, Any]],
    windows: Sequence[WindowSpec],
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for spec, raw in zip(windows, raw_results):
        eid = int(raw.get("emotion_id", 4))
        formatted.append(
            {
                "index": spec.index,
                "start_sec": round(spec.start_sec, 2),
                "end_sec": round(spec.end_sec, 2),
                "emotion_id": eid,
                "emotion_label": raw.get("emotion_label", EMOTION_NAMES[eid] if eid < len(EMOTION_NAMES) else "other"),
                "emotion_label_cn": EMOTION_NAMES_CN[eid] if eid < len(EMOTION_NAMES_CN) else "其他",
                "confidence": float(raw.get("confidence", 0.0)),
                "valence": float(raw.get("valence", 0.0)),
                "arousal": float(raw.get("arousal", 0.0)),
                "all_probs": raw.get("all_probs") or [],
            }
        )
    return formatted

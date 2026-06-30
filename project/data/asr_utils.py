"""离线 ASR：为 CREMA 等数据集从音频生成训练用文本。"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np

# CREMA-D 文件名第二段 → 12 句标准台词（README 官方映射，ASR 失败时回退）
CREMA_SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

_WHISPER_MODEL = None
_WHISPER_PROCESSOR = None
_FASTER_WHISPER = None


def is_crema_placeholder_text(text: str) -> bool:
    t = (text or "").strip()
    return (not t) or t.startswith("Audio transcription for")


def crema_sentence_from_stem(stem: str) -> Optional[str]:
    """从原始 CREMA 文件名 stem（如 1001_DFA_ANG_XX）解析台词。"""
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    code = parts[1].upper()
    return CREMA_SENTENCE_MAP.get(code)


def _load_mono_float(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    import librosa

    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return y.astype(np.float32)


def _transcribe_faster_whisper(audio_path: str, model_size: str, device: str) -> str:
    global _FASTER_WHISPER
    from faster_whisper import WhisperModel

    if _FASTER_WHISPER is None:
        compute_type = "float16" if device.startswith("cuda") else "int8"
        _FASTER_WHISPER = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = _FASTER_WHISPER.transcribe(audio_path, language="en", beam_size=1)
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).strip()


def _transcribe_transformers_whisper(
    audio_path: str, model_id: str, device: str, sample_rate: int = 16000
) -> str:
    global _WHISPER_MODEL, _WHISPER_PROCESSOR
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if _WHISPER_MODEL is None:
        _WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(model_id)
        _WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained(model_id)
        _WHISPER_MODEL.to(device)
        _WHISPER_MODEL.eval()

    waveform = _load_mono_float(audio_path, sample_rate)
    inputs = _WHISPER_PROCESSOR(
        waveform, sampling_rate=sample_rate, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        ids = _WHISPER_MODEL.generate(input_features, max_new_tokens=128)
    text = _WHISPER_PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_audio_file(
    audio_path: str,
    *,
    engine: str = "auto",
    model: str = "base",
    device: str = "cpu",
    sample_rate: int = 16000,
) -> Tuple[str, str]:
    """
    转写单个音频文件。

    Returns:
        (text, engine_used)
    """
    if not os.path.isfile(audio_path):
        return "", "missing"

    engines = []
    if engine == "auto":
        engines = ["transformers", "faster_whisper"]
    else:
        engines = [engine]

    last_err = None
    for eng in engines:
        try:
            if eng == "faster_whisper":
                text = _transcribe_faster_whisper(audio_path, model, device)
                return text, eng
            if eng == "transformers":
                model_id = model if "/" in model else f"openai/whisper-{model}"
                text = _transcribe_transformers_whisper(
                    audio_path, model_id, device, sample_rate
                )
                return text, eng
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"ASR failed for {audio_path}: {last_err}")


def normalize_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text

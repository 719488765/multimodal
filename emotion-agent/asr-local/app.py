from __future__ import annotations

import asyncio
import os
import tempfile
import time
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import wave
import numpy as np

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover
    WhisperModel = None  # type: ignore


APP_NAME = "asr-local-whisper"


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


MODEL_NAME = _env("WHISPER_MODEL", "small")
DEVICE = _env("WHISPER_DEVICE", "cuda")  # cuda|cpu|auto
DEVICE_INDEX = int(_env("WHISPER_DEVICE_INDEX", "0"))
COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "float16")  # float16|int8_float16|int8
BEAM_SIZE = int(_env("WHISPER_BEAM_SIZE", "5"))
FAST_BEAM_SIZE = int(_env("WHISPER_FAST_BEAM_SIZE", "3"))
SHORT_AUDIO_SEC = float(_env("WHISPER_SHORT_AUDIO_SEC", "8.0"))
CPU_THREADS = int(_env("WHISPER_CPU_THREADS", "4"))
SILENCE_RMS = float(_env("WHISPER_SILENCE_RMS", "0.008"))

_model: Optional["WhisperModel"] = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_model() -> "WhisperModel":
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt")
        device = DEVICE
        if device == "auto":
            device = "cuda"
        kwargs: Dict[str, Any] = {
            "device": device,
            "compute_type": COMPUTE_TYPE,
            "cpu_threads": CPU_THREADS,
        }
        if device == "cuda":
            kwargs["device_index"] = DEVICE_INDEX
        print(f"[ASR] loading model={MODEL_NAME} device={device} index={DEVICE_INDEX} compute={COMPUTE_TYPE}")
        _model = WhisperModel(MODEL_NAME, **kwargs)
        return _model


def _trim_silence_edges(audio: np.ndarray, sample_rate: int, threshold: float = SILENCE_RMS) -> np.ndarray:
    """去掉首尾静音，避免 Whisper 在空白段幻觉或漏掉有效语音。"""
    if audio.size == 0:
        return audio
    window = max(1, int(sample_rate * 0.02))
    n = audio.size
    start = 0
    for i in range(0, n, window):
        chunk = audio[i : i + window]
        if float(np.sqrt(np.mean(np.square(chunk)))) >= threshold:
            start = i
            break
    end = n
    for i in range(n - window, -1, -window):
        chunk = audio[i : i + window]
        if float(np.sqrt(np.mean(np.square(chunk)))) >= threshold:
            end = min(n, i + window)
            break
    if end <= start:
        return audio
    return audio[start:end]


def _decode_text_with_fallback(model: "WhisperModel", audio_input, language_hint: str, *, fast: bool = False):
    """
    First try with language hint and stable params.
    If empty, retry with looser params to reduce empty-text cases.
    """
    beam = FAST_BEAM_SIZE if fast else BEAM_SIZE
    transcribe_kwargs = dict(
        language=language_hint or None,
        vad_filter=False,
        beam_size=beam,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.5,
    )
    with _infer_lock:
        segments, info = model.transcribe(audio_input, **transcribe_kwargs)
    text = "".join(s.text for s in segments).strip()
    if text or fast:
        return text, info

    # Fallback pass: remove language constraint to avoid wrong language lock.
    try:
        with _infer_lock:
            segments2, info2 = model.transcribe(
                audio_input,
                language=None,
                vad_filter=False,
                beam_size=beam,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=True,
            )
        text2 = "".join(s.text for s in segments2).strip()
        return text2, info2
    except Exception:
        return "", info


app = FastAPI(title=APP_NAME)


@app.on_event("startup")
def warmup_model() -> None:
    """后台预加载模型，不阻塞 /health 与端口监听。"""

    def _run() -> None:
        try:
            get_model()
            print(f"[ASR] warmup ok model={MODEL_NAME} device={DEVICE} beam={BEAM_SIZE}")
        except Exception as exc:
            print(f"[ASR] warmup failed: {exc}")

    threading.Thread(target=_run, daemon=True, name="asr-warmup").start()


def _transcribe_payload(content: bytes, suffix: str, language: str) -> Dict[str, Any]:
    """CPU/GPU 密集转写逻辑，须在 worker 线程中执行。"""
    t0 = time.time()
    tmp_path: Path | None = None
    try:
        m = get_model()
        lang = language or "zh"

        if suffix == ".wav":
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(content)
                tmp_path = Path(f.name)
            with wave.open(str(tmp_path), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
            if sampwidth != 2:
                raise RuntimeError(f"Only 16-bit PCM WAV is supported, got sampwidth={sampwidth}")
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            if audio.size == 0:
                return {
                    "text": "",
                    "language": lang,
                    "duration": 0,
                    "provider": "whisper_local",
                    "model": MODEL_NAME,
                    "latency_ms": int((time.time() - t0) * 1000),
                }
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))
            if rms < 0.002 and peak < 0.02:
                print(
                    f"[ASR] suffix=.wav bytes={len(content)} duration~={audio.size/16000:.2f}s "
                    f"low_energy rms={rms:.6f} peak={peak:.6f}"
                )
                return {
                    "text": "",
                    "language": lang,
                    "duration": round(audio.size / 16000, 3),
                    "provider": "whisper_local",
                    "model": MODEL_NAME,
                    "latency_ms": int((time.time() - t0) * 1000),
                }
            duration_sec = audio.size / max(sr, 1)
            fast = duration_sec <= SHORT_AUDIO_SEC
            audio = _trim_silence_edges(audio, sr)
            text, info = _decode_text_with_fallback(m, audio, lang, fast=fast)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(content)
                tmp_path = Path(f.name)
            text, info = _decode_text_with_fallback(m, str(tmp_path), lang, fast=True)

        dur = getattr(info, "duration", None)
        print(f"[ASR] suffix={suffix} bytes={len(content)} duration={dur} text_len={len(text)}")
        return {
            "text": text,
            "language": getattr(info, "language", lang),
            "duration": dur,
            "provider": "whisper_local",
            "model": MODEL_NAME,
            "latency_ms": int((time.time() - t0) * 1000),
        }
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": APP_NAME,
        "model": MODEL_NAME,
        "device": DEVICE,
        "device_index": DEVICE_INDEX,
        "compute_type": COMPUTE_TYPE,
        "model_loaded": _model is not None,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: str = Form("zh"),
):
    """OpenAI-compatible transcription endpoint."""
    suffix = Path(file.filename or "").suffix.lower() or ".bin"
    content = await file.read()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _transcribe_payload, content, suffix, language or "zh")
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {
                "text": "",
                "error": str(e),
                "provider": "whisper_local",
            },
            status_code=500,
        )


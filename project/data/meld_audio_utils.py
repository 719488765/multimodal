"""MELD 视频 → 训练用 WAV 音频提取（mono 16kHz，与 config data.audio 对齐）"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import wave
from typing import Optional, Tuple

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


def pyav_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


def ffmpeg_available() -> bool:
    if shutil.which("ffmpeg"):
        return True
    conda_bin = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    return os.path.isfile(conda_bin) and os.access(conda_bin, os.X_OK)


def meld_wav_path_for_video(video_path: str, audio_dir: Optional[str] = None) -> str:
    """根据 meld_* .mp4 路径推断对应 .wav 路径。"""
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = audio_dir or os.path.join(os.path.dirname(os.path.dirname(video_path)), "audio")
    return os.path.join(out_dir, f"{base}.wav")


def _frame_to_mono_float(frame) -> np.ndarray:
    """任意声道数 → mono float32（planar: (C, N) 或 packed）。"""
    arr = frame.to_ndarray()
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        return arr.mean(axis=0).astype(np.float32)
    return arr.mean(axis=1).astype(np.float32)


def _write_pcm_wav(wav_path: str, samples: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def extract_audio_pyav(
    video_path: str,
    wav_path: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> Tuple[bool, str]:
    import av
    import librosa

    if not os.path.isfile(video_path):
        return False, f"missing video: {video_path}"

    try:
        container = av.open(video_path)
    except Exception as exc:
        return False, f"av.open failed: {exc!r}"

    if not any(s.type == "audio" for s in container.streams):
        container.close()
        return False, "no audio stream"

    chunks: list[np.ndarray] = []
    src_rate: Optional[int] = None

    try:
        for frame in container.decode(audio=0):
            src_rate = int(frame.sample_rate or src_rate or sample_rate)
            chunks.append(_frame_to_mono_float(frame))
    except Exception as exc:
        container.close()
        return False, f"decode failed: {exc!r}"
    finally:
        container.close()

    if not chunks or src_rate is None:
        return False, "no audio decoded"

    audio = np.concatenate(chunks).astype(np.float32)
    if audio.size == 0:
        return False, "empty audio"

    if src_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=src_rate, target_sr=sample_rate)

    if channels != 1:
        return False, f"only mono output supported, got channels={channels}"

    _write_pcm_wav(wav_path, audio, sample_rate, channels=1)
    return True, "ok"


def extract_audio_ffmpeg(
    video_path: str,
    wav_path: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    overwrite: bool = False,
    timeout_sec: int = 120,
) -> Tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg") or os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if not ffmpeg or not os.path.isfile(ffmpeg):
        return False, "ffmpeg not found"

    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        wav_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_sec}s"

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or f"ffmpeg exit {proc.returncode}"

    if not os.path.isfile(wav_path) or os.path.getsize(wav_path) < 44:
        return False, "output wav missing or empty"

    return True, "ok"


def extract_audio_from_video(
    video_path: str,
    wav_path: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    overwrite: bool = False,
    timeout_sec: int = 120,
) -> Tuple[bool, str]:
    """
    从 mp4 提取 mono WAV。优先 PyAV + librosa（支持 6 声道等），回退 ffmpeg CLI。
    """
    if not os.path.isfile(video_path):
        return False, f"missing video: {video_path}"

    if os.path.isfile(wav_path) and not overwrite and os.path.getsize(wav_path) > 44:
        return True, "skipped existing"

    pyav_msg = ""
    if pyav_available():
        ok, pyav_msg = extract_audio_pyav(
            video_path, wav_path, sample_rate=sample_rate, channels=channels
        )
        if ok:
            return ok, pyav_msg

    if ffmpeg_available():
        ok, ff_msg = extract_audio_ffmpeg(
            video_path,
            wav_path,
            sample_rate=sample_rate,
            channels=channels,
            overwrite=True,
            timeout_sec=timeout_sec,
        )
        if ok:
            return ok, ff_msg
        return False, ff_msg

    if pyav_msg:
        return False, pyav_msg
    return False, "no backend: install PyAV (pip install av) or ffmpeg"


def _worker_extract(args: Tuple[str, str, int, bool]) -> Tuple[str, bool, str]:
    video_path, wav_path, sample_rate, overwrite = args
    ok, msg = extract_audio_from_video(
        video_path, wav_path, sample_rate=sample_rate, overwrite=overwrite
    )
    return video_path, ok, msg

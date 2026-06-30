"""
在线/CLI 共用的多模态情绪推理运行时。
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover
    librosa = None

try:
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover
    sf = None

from transformers import BertTokenizer  # type: ignore

from models import MultimodalEmotionModel
from utils import load_checkpoint, load_config, setup_device
from utils.temporal_inference import (
    aggregate_window_predictions,
    audio_duration_sec,
    build_temporal_summary,
    compute_windows,
    format_window_results,
    load_audio_waveform,
    merge_temporal_config,
    should_use_temporal,
    slice_audio_window_tensor,
)
from utils.video_frame_utils import (
    frames_list_to_tensor,
    is_image_mime,
    is_video_mime,
    preprocess_video_from_file_bytes,
    preprocess_video_from_file_path,
    preprocess_video_window_from_bytes,
    suffix_for_mime,
)

_frames_list_to_tensor = frames_list_to_tensor

logger = logging.getLogger(__name__)

EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]
EMOTION_NAMES_CN = ["开心", "难过", "生气", "害怕", "平静", "焦虑", "其他"]

_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,(.+)$", re.DOTALL | re.IGNORECASE)


def strip_base64_payload(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    match = _DATA_URL_RE.match(value)
    if match:
        return match.group(1).strip()
    return value


def decode_base64_bytes(value: str) -> bytes:
    payload = strip_base64_payload(value)
    if not payload:
        return b""
    padding = (-len(payload)) % 4
    if padding:
        payload += "=" * padding
    return base64.b64decode(payload)


def preprocess_video_from_path(video_path: str, config: dict) -> Optional[torch.Tensor]:
    return preprocess_video_from_file_path(video_path, config)


def preprocess_video_from_bytes(
    video_bytes: bytes,
    config: dict,
    *,
    video_mime: str = "",
    video_filename: str = "",
) -> Tuple[Optional[torch.Tensor], str, int]:
    """返回 (tensor, decode_mode, frames_extracted)。"""
    if not video_bytes:
        return None, "empty", 0

    video_config = config.get("data", {}).get("video", {})
    num_frames = video_config.get("num_frames", 4)

    if is_video_mime(video_mime) or suffix_for_mime(video_mime, video_filename) in (".webm", ".mp4", ".mov"):
        suffix = suffix_for_mime(video_mime, video_filename)
        tensor = preprocess_video_from_file_bytes(video_bytes, config, suffix=suffix)
        if tensor is not None:
            return tensor, "video_file", num_frames

    frame_size = video_config.get("frame_size", 112)
    if cv2 is not None:
        arr = np.frombuffer(video_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.resize(frame, (frame_size, frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return _frames_list_to_tensor([frame], num_frames), "single_frame_fallback", 1

    image = Image.open(io.BytesIO(video_bytes)).convert("RGB")
    frame = np.array(image.resize((frame_size, frame_size)), dtype=np.uint8)
    return _frames_list_to_tensor([frame], num_frames), "single_frame_fallback", 1


def preprocess_video_from_image_bytes(image_bytes: bytes, config: dict) -> Optional[torch.Tensor]:
    tensor, mode, _ = preprocess_video_from_bytes(image_bytes, config)
    return tensor if mode != "empty" else None


def preprocess_audio_from_path(audio_path: str, config: dict) -> Optional[torch.Tensor]:
    if librosa is None:
        raise RuntimeError("librosa is required for audio preprocessing")
    audio_config = config["data"]["audio"]
    sample_rate = audio_config.get("sample_rate", 16000)
    duration = audio_config.get("duration", 3.0)
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return _waveform_to_tensor(audio, sample_rate, duration)
    except Exception:
        return None


def preprocess_audio_from_bytes(audio_bytes: bytes, config: dict) -> Optional[torch.Tensor]:
    if not audio_bytes:
        return None
    audio_config = config["data"]["audio"]
    sample_rate = audio_config.get("sample_rate", 16000)
    duration = audio_config.get("duration", 3.0)

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
    return _waveform_to_tensor(np.asarray(audio, dtype=np.float32), sample_rate, duration)


def _waveform_to_tensor(audio: np.ndarray, sample_rate: int, duration: float) -> torch.Tensor:
    target_length = int(sample_rate * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    elif len(audio) > target_length:
        # 与在线采集一致：取最近 duration 秒（尾部对齐 ASR / 用户停止前的发言）
        audio = audio[-target_length:]
    return torch.from_numpy(audio).float().unsqueeze(0)


def preprocess_physiological_from_path(physiological_path: str) -> Optional[torch.Tensor]:
    try:
        data = np.load(physiological_path)
        return torch.from_numpy(data).float().unsqueeze(0)
    except Exception:
        return None


class EmotionInferenceService:
    """加载 checkpoint 一次，供 agent 与 CLI 复用。"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.project_root = project_root
        self._device_override = device
        self.config: Optional[dict] = None
        self.device: Optional[torch.device] = None
        self.model: Optional[MultimodalEmotionModel] = None
        self.tokenizer: Optional[BertTokenizer] = None
        self._loaded = False
        self._temporal_cfg: Dict[str, Any] = merge_temporal_config()

    def load(self) -> None:
        import os

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.config = load_config(self.config_path)
        if self._device_override:
            self.config["device"] = self._device_override
        self._temporal_cfg = merge_temporal_config(self.config.get("temporal_inference"))
        self.device = setup_device(self.config)

        self.model = MultimodalEmotionModel(self.config).to(self.device)
        load_checkpoint(self.checkpoint_path, self.model)
        self.model.eval()

        text_backbone = self.config["model"]["text"]["backbone"]
        self.tokenizer = BertTokenizer.from_pretrained(text_backbone)
        self._loaded = True
        fusion = self.config.get("model", {}).get("attention", {}).get("fusion_strategy", "unknown")
        logger.info(
            "[TRAINED_MODEL] loaded checkpoint=%s fusion=%s device=%s",
            os.path.basename(self.checkpoint_path),
            fusion,
            self.device,
        )

    def _preprocess_text(self, text: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not text or not self.tokenizer:
            return None, None
        encoded = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def predict_from_paths(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        physiological_path: Optional[str] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._loaded or self.model is None or self.config is None or self.device is None:
            raise RuntimeError("EmotionInferenceService.load() must be called first")

        video = preprocess_video_from_path(video_path, self.config) if video_path else None
        audio = preprocess_audio_from_path(audio_path, self.config) if audio_path else None
        physiological = (
            preprocess_physiological_from_path(physiological_path) if physiological_path else None
        )
        text_ids, text_mask = self._preprocess_text(text or "")

        return self._forward(
            video=video,
            audio=audio,
            physiological=physiological,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            has_video=bool(video_path),
            has_audio=bool(audio_path),
            has_text=bool(text and text.strip()),
        )

    def set_temporal_config(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        self._temporal_cfg = merge_temporal_config(overrides)

    def predict_from_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded or self.config is None:
            raise RuntimeError("EmotionInferenceService.load() must be called first")

        metadata = sample.get("metadata") or {}
        temporal_cfg = merge_temporal_config(
            {**self._temporal_cfg, **metadata.get("temporal_inference", {})}
        )
        audio_b64 = sample.get("audio_chunk_b64") or ""
        if temporal_cfg.get("enabled") and audio_b64:
            raw = decode_base64_bytes(audio_b64)
            audio_cfg = self.config.get("data", {}).get("audio", {})
            sr = audio_cfg.get("sample_rate", 16000)
            waveform = load_audio_waveform(raw, sample_rate=sr)
            total_sec = audio_duration_sec(waveform, sr) if waveform is not None else 0.0
            window_sec = float(temporal_cfg.get("window_sec", 3.0))
            margin = float(temporal_cfg.get("short_path_margin_sec", 0.2))
            use_temporal = should_use_temporal(total_sec, window_sec, margin, enabled=True)
            frames_b64 = metadata.get("capture_frames_b64") or []
            video_mime = str(metadata.get("video_mime") or "")
            is_static_jpeg = (
                bool(frames_b64)
                or "jpeg" in video_mime.lower()
                or str(metadata.get("video_filename") or "").lower().endswith(".jpg")
            )
            short_max = float(temporal_cfg.get("short_path_max_sec", 6.0))
            if (
                use_temporal
                and temporal_cfg.get("jpeg_prefer_single_window", True)
                and is_static_jpeg
                and total_sec <= short_max
            ):
                use_temporal = False
            if use_temporal:
                return self.predict_from_sample_temporal(sample, temporal_cfg)

        return self._predict_from_sample_single(sample)

    def _predict_from_sample_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loaded or self.config is None:
            raise RuntimeError("EmotionInferenceService.load() must be called first")

        t0 = time.perf_counter()
        video_b64 = sample.get("video_chunk_b64") or ""
        audio_b64 = sample.get("audio_chunk_b64") or ""
        text = (sample.get("text") or "").strip()

        video = None
        audio = None
        video_decode_mode = "empty"
        frames_extracted = 0
        metadata = sample.get("metadata") or {}
        video_mime = str(metadata.get("video_mime") or "")
        video_filename = str(metadata.get("video_filename") or "")

        capture_frames: List[str] = list(metadata.get("capture_frames_b64") or [])
        if capture_frames:
            raw_frame = decode_base64_bytes(capture_frames[-1])
            video, video_decode_mode, frames_extracted = preprocess_video_from_bytes(
                raw_frame,
                self.config,
                video_mime="image/jpeg",
                video_filename="capture_last.jpg",
            )
            video_decode_mode = "multi_frame_sequence"
        elif video_b64:
            raw_video = decode_base64_bytes(video_b64)
            video, video_decode_mode, frames_extracted = preprocess_video_from_bytes(
                raw_video,
                self.config,
                video_mime=video_mime,
                video_filename=video_filename,
            )
        if audio_b64:
            audio = preprocess_audio_from_bytes(decode_base64_bytes(audio_b64), self.config)

        text_ids, text_mask = self._preprocess_text(text)

        modalities = self.config["model"].get("modalities", {})
        use_video = modalities.get("use_video", True)
        use_audio = modalities.get("use_audio", True)
        use_text = modalities.get("use_text", True)

        video_bytes = len(decode_base64_bytes(video_b64)) if video_b64 else 0
        audio_bytes = len(decode_base64_bytes(audio_b64)) if audio_b64 else 0

        degraded = (
            (use_video and video is None)
            or (use_audio and audio is None)
            or (use_text and not text)
        )

        text_backbone = self.config.get("model", {}).get("text", {}).get("backbone", "bert-base-uncased")
        video_cfg = self.config.get("data", {}).get("video", {})
        audio_cfg = self.config.get("data", {}).get("audio", {})

        modality_trace = {
            "video": {
                "enabled": use_video,
                "received_bytes": video_bytes,
                "preprocessed": video is not None,
                "tensor_shape": list(video.shape) if video is not None else None,
                "num_frames": video_cfg.get("num_frames", 4),
                "frame_size": video_cfg.get("frame_size", 112),
                "source": video_mime or "browser_capture",
                "decode_mode": video_decode_mode,
                "frames_extracted": frames_extracted,
                "matched_training_pipeline": video_decode_mode == "video_file",
            },
            "audio": {
                "enabled": use_audio,
                "received_bytes": audio_bytes,
                "preprocessed": audio is not None,
                "tensor_shape": list(audio.shape) if audio is not None else None,
                "sample_rate": audio_cfg.get("sample_rate", 16000),
                "duration_sec": audio_cfg.get("duration", 3.0),
            },
            "text": {
                "enabled": use_text,
                "content_preview": (text[:80] + "…") if len(text) > 80 else text,
                "char_len": len(text),
                "preprocessed": text_ids is not None,
                "tokenizer": text_backbone,
                "token_count": int(text_ids.shape[1]) if text_ids is not None else 0,
            },
        }

        result = self._forward(
            video=video,
            audio=audio,
            physiological=None,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            has_video=video is not None,
            has_audio=audio is not None,
            has_text=bool(text),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result["degraded_mode"] = degraded
        result["inference_source"] = "checkpoint"
        result["inference_ms"] = round(elapsed_ms, 2)
        result["fusion_strategy"] = (
            self.config.get("model", {}).get("attention", {}).get("fusion_strategy", "")
        )
        result["checkpoint_file"] = os.path.basename(self.checkpoint_path)
        probs = result.get("all_probs") or []
        result["all_probs_labeled"] = [
            {
                "id": i,
                "label": EMOTION_NAMES[i],
                "label_cn": EMOTION_NAMES_CN[i],
                "prob": float(probs[i]) if i < len(probs) else 0.0,
            }
            for i in range(len(EMOTION_NAMES))
        ]
        result["pipeline_trace"] = {
            "modalities": modality_trace,
            "model": {
                "called": True,
                "is_mock": False,
                "inference_source": "checkpoint",
                "checkpoint_file": result["checkpoint_file"],
                "fusion_strategy": result["fusion_strategy"],
                "device": str(self.device),
                "inference_ms": result["inference_ms"],
                "modalities_config": {
                    "use_video": use_video,
                    "use_audio": use_audio,
                    "use_text": use_text,
                },
            },
            "output": {
                "emotion_id": result.get("emotion_id"),
                "emotion_label": result.get("emotion_label"),
                "confidence": result.get("confidence"),
                "degraded_mode": degraded,
                "all_probs_labeled": result["all_probs_labeled"],
            },
            "notes": [
                "已调用训练 checkpoint 前向推理（非 mock）",
                "文本经 ASR 或用户输入合并后送入 BERT",
                f"视频解码模式: {video_decode_mode}（video_file=与训练一致抽帧）",
                "若 decode_mode=single_frame_fallback，建议改用浏览器 webm 短视频 clip",
            ],
        }
        logger.info(
            "[TRAINED_MODEL] forward label=%s id=%s conf=%.3f ms=%.1f degraded=%s probs_top=%s",
            result.get("emotion_label"),
            result.get("emotion_id"),
            float(result.get("confidence") or 0),
            elapsed_ms,
            degraded,
            self._format_top_probs(result.get("all_probs") or []),
        )
        return result

    def predict_from_sample_temporal(
        self,
        sample: Dict[str, Any],
        temporal_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._loaded or self.config is None:
            raise RuntimeError("EmotionInferenceService.load() must be called first")

        cfg = merge_temporal_config(temporal_cfg or self._temporal_cfg)
        t0 = time.perf_counter()

        video_b64 = sample.get("video_chunk_b64") or ""
        audio_b64 = sample.get("audio_chunk_b64") or ""
        text = (sample.get("text") or "").strip()
        metadata = sample.get("metadata") or {}
        video_mime = str(metadata.get("video_mime") or "")
        video_filename = str(metadata.get("video_filename") or "")

        audio_cfg = self.config.get("data", {}).get("audio", {})
        sample_rate = audio_cfg.get("sample_rate", 16000)
        window_sec = float(cfg.get("window_sec", 3.0))
        stride_sec = float(cfg.get("stride_sec", 3.0))
        max_windows = int(cfg.get("max_windows", 10))

        raw_audio = decode_base64_bytes(audio_b64) if audio_b64 else b""
        waveform = load_audio_waveform(raw_audio, sample_rate=sample_rate)
        total_sec = audio_duration_sec(waveform, sample_rate) if waveform is not None else 0.0
        if total_sec <= 0:
            return self._predict_from_sample_single(sample)

        windows = compute_windows(total_sec, window_sec, stride_sec, max_windows)
        raw_video = decode_base64_bytes(video_b64) if video_b64 else b""
        video_suffix = suffix_for_mime(video_mime, video_filename)
        is_video_file = bool(raw_video) and (
            is_video_mime(video_mime) or video_suffix in (".webm", ".mp4", ".mov")
        )

        capture_frames: List[str] = list(metadata.get("capture_frames_b64") or [])
        video_tensors: List[torch.Tensor] = []
        audio_tensors: List[torch.Tensor] = []
        for spec in windows:
            if waveform is not None:
                audio_tensors.append(
                    slice_audio_window_tensor(waveform, sample_rate, spec.start_sec, window_sec).unsqueeze(0)
                )
            vt = None
            if capture_frames:
                frame_idx = min(spec.index, len(capture_frames) - 1)
                frame_raw = decode_base64_bytes(capture_frames[frame_idx])
                if frame_raw:
                    vt, _, _ = preprocess_video_from_bytes(
                        frame_raw,
                        self.config,
                        video_mime="image/jpeg",
                        video_filename=f"capture_frame_{frame_idx}.jpg",
                    )
            elif is_video_file and raw_video:
                vt = preprocess_video_window_from_bytes(
                    raw_video,
                    self.config,
                    offset_sec=spec.start_sec,
                    duration_sec=window_sec,
                    suffix=video_suffix,
                )
            elif raw_video:
                vt, _, _ = preprocess_video_from_bytes(
                    raw_video,
                    self.config,
                    video_mime=video_mime,
                    video_filename=video_filename,
                )
            if vt is not None:
                video_tensors.append(vt)

        n = len(windows)
        if not audio_tensors:
            return self._predict_from_sample_single(sample)

        audio_batch = torch.cat(audio_tensors, dim=0)
        video_batch = None
        if video_tensors:
            while len(video_tensors) < n:
                video_tensors.append(video_tensors[-1])
            video_batch = torch.cat(video_tensors[:n], dim=0)

        text_ids, text_mask = self._preprocess_text(text)
        if text_ids is not None and text_mask is not None:
            text_ids = text_ids.repeat(n, 1)
            text_mask = text_mask.repeat(n, 1)

        modalities = self.config["model"].get("modalities", {})
        use_video = modalities.get("use_video", True)
        use_audio = modalities.get("use_audio", True)
        use_text = modalities.get("use_text", True)

        batch_results = self._forward_batch(
            video=video_batch if use_video else None,
            audio=audio_batch if use_audio else None,
            physiological=None,
            text_input_ids=text_ids if use_text else None,
            text_attention_mask=text_mask if use_text else None,
            has_video=video_batch is not None,
            has_audio=True,
            has_text=bool(text),
        )

        aggregated = aggregate_window_predictions(
            batch_results,
            strategy=str(cfg.get("aggregation", "recency_weighted")),
            recency_alpha=float(cfg.get("recency_alpha", 1.5)),
        )
        temporal_windows = format_window_results(batch_results, windows)
        temporal_summary = build_temporal_summary(
            batch_results,
            windows,
            str(cfg.get("aggregation", "recency_weighted")),
            total_sec,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        probs = aggregated.get("all_probs") or []
        video_bytes = len(raw_video)
        audio_bytes = len(raw_audio)
        if capture_frames:
            video_decode_mode = "multi_frame_sequence"
        elif is_video_file:
            video_decode_mode = "video_file"
        elif raw_video:
            video_decode_mode = "single_frame_fallback"
        else:
            video_decode_mode = "empty"
        text_backbone = self.config.get("model", {}).get("text", {}).get("backbone", "bert-base-uncased")
        video_cfg = self.config.get("data", {}).get("video", {})
        num_frames = video_cfg.get("num_frames", 4)

        degraded = (
            (use_video and video_batch is None)
            or (use_audio and audio_batch is None)
            or (use_text and not text)
        )

        modality_trace = {
            "video": {
                "enabled": use_video,
                "received_bytes": video_bytes,
                "preprocessed": video_batch is not None,
                "tensor_shape": list(video_batch.shape) if video_batch is not None else None,
                "num_frames": num_frames,
                "frame_size": video_cfg.get("frame_size", 112),
                "source": video_mime or "browser_capture",
                "decode_mode": video_decode_mode,
                "frames_extracted": num_frames * n if video_batch is not None else 0,
                "capture_frame_count": len(capture_frames),
                "matched_training_pipeline": video_decode_mode == "video_file",
                "temporal_windows": n,
            },
            "audio": {
                "enabled": use_audio,
                "received_bytes": audio_bytes,
                "preprocessed": audio_batch is not None,
                "tensor_shape": list(audio_batch.shape) if audio_batch is not None else None,
                "sample_rate": sample_rate,
                "duration_sec": round(total_sec, 2),
                "window_sec": window_sec,
                "temporal_windows": n,
            },
            "text": {
                "enabled": use_text,
                "content_preview": (text[:80] + "…") if len(text) > 80 else text,
                "char_len": len(text),
                "preprocessed": text_ids is not None,
                "tokenizer": text_backbone,
                "token_count": int(text_ids.shape[1]) if text_ids is not None else 0,
                "note": "各时间窗共用全量 ASR 文本",
            },
        }
        all_probs_labeled = [
            {
                "id": i,
                "label": EMOTION_NAMES[i],
                "label_cn": EMOTION_NAMES_CN[i],
                "prob": float(probs[i]) if i < len(probs) else 0.0,
            }
            for i in range(len(EMOTION_NAMES))
        ]

        per_window_probs = [
            {"index": w.get("index"), "label": w.get("emotion_label"), "probs": w.get("all_probs")}
            for w in temporal_windows
        ]
        result = {
            **aggregated,
            "per_window_probs": per_window_probs,
            "video_decode_mode": video_decode_mode,
            "degraded_mode": degraded,
            "inference_source": "checkpoint_temporal",
            "inference_ms": round(elapsed_ms, 2),
            "fusion_strategy": self.config.get("model", {}).get("attention", {}).get("fusion_strategy", ""),
            "checkpoint_file": os.path.basename(self.checkpoint_path),
            "temporal_windows": temporal_windows,
            "temporal_summary": temporal_summary,
            "all_probs_labeled": all_probs_labeled,
            "pipeline_trace": {
                "temporal": {
                    "enabled": True,
                    "num_windows": n,
                    "total_duration_sec": round(total_sec, 2),
                    "window_sec": window_sec,
                    "stride_sec": stride_sec,
                    "aggregation": cfg.get("aggregation"),
                    "batch_inference": bool(cfg.get("batch_windows", True)),
                    "windows_preview": [
                        {"start": w["start_sec"], "end": w["end_sec"], "label": w["emotion_label"]}
                        for w in temporal_windows[:5]
                    ],
                },
                "modalities": modality_trace,
                "model": {
                    "called": True,
                    "is_mock": False,
                    "inference_source": "checkpoint_temporal",
                    "checkpoint_file": os.path.basename(self.checkpoint_path),
                    "fusion_strategy": self.config.get("model", {}).get("attention", {}).get("fusion_strategy", ""),
                    "device": str(self.device),
                    "inference_ms": round(elapsed_ms, 2),
                    "batch_size": n,
                    "modalities_config": {
                        "use_video": use_video,
                        "use_audio": use_audio,
                        "use_text": use_text,
                    },
                },
                "output": {
                    "emotion_id": aggregated.get("emotion_id"),
                    "emotion_label": aggregated.get("emotion_label"),
                    "confidence": aggregated.get("confidence"),
                    "degraded_mode": degraded,
                    "all_probs_labeled": all_probs_labeled,
                    "aggregation": cfg.get("aggregation"),
                },
                "notes": [
                    f"长时推理：{n} 个 {window_sec}s 窗口，近端加权聚合",
                    f"视频解码: {video_decode_mode}",
                    "文本经 ASR 或用户输入合并后各窗共用",
                ],
            },
        }
        logger.info(
            "[TRAINED_MODEL] temporal n=%d total=%.1fs label=%s conf=%.3f ms=%.1f",
            n,
            total_sec,
            result.get("emotion_label"),
            float(result.get("confidence") or 0),
            elapsed_ms,
        )
        return result

    @staticmethod
    def _format_top_probs(probs: list, k: int = 3) -> str:
        if not probs:
            return "[]"
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:k]
        return ",".join(f"{EMOTION_NAMES[i]}:{p:.3f}" for i, p in indexed)

    def _forward(
        self,
        video: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
        physiological: Optional[torch.Tensor],
        text_input_ids: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        has_video: bool,
        has_audio: bool,
        has_text: bool,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.device is not None

        if video is not None:
            video = video.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)
        if physiological is not None:
            physiological = physiological.to(self.device)
        if text_input_ids is not None:
            text_input_ids = text_input_ids.to(self.device)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                video=video,
                audio=audio,
                physiological=physiological,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
            )

        emotion_probs = outputs["emotion_probs"].detach().cpu().numpy()[0]
        emotion_id = int(np.argmax(emotion_probs))
        emotion_dimensions = outputs["emotion_dimensions"].detach().cpu().numpy()[0]

        degraded = not (has_video and has_audio and has_text)

        return {
            "emotion_label": EMOTION_NAMES[emotion_id],
            "emotion": EMOTION_NAMES[emotion_id],
            "emotion_id": emotion_id,
            "confidence": float(emotion_probs[emotion_id]),
            "valence": float(emotion_dimensions[0]),
            "arousal": float(emotion_dimensions[1]),
            "all_probs": emotion_probs.tolist(),
            "degraded_mode": degraded,
        }

    def _forward_batch(
        self,
        video: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
        physiological: Optional[torch.Tensor],
        text_input_ids: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        has_video: bool,
        has_audio: bool,
        has_text: bool,
    ) -> List[Dict[str, Any]]:
        assert self.model is not None and self.device is not None

        if video is not None:
            video = video.to(self.device)
        if audio is not None:
            audio = audio.to(self.device)
        if physiological is not None:
            physiological = physiological.to(self.device)
        if text_input_ids is not None:
            text_input_ids = text_input_ids.to(self.device)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                video=video,
                audio=audio,
                physiological=physiological,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
            )

        probs_batch = outputs["emotion_probs"].detach().cpu().numpy()
        dims_batch = outputs["emotion_dimensions"].detach().cpu().numpy()
        degraded = not (has_video and has_audio and has_text)

        results: List[Dict[str, Any]] = []
        for i in range(probs_batch.shape[0]):
            emotion_probs = probs_batch[i]
            emotion_id = int(np.argmax(emotion_probs))
            emotion_dimensions = dims_batch[i]
            results.append(
                {
                    "emotion_label": EMOTION_NAMES[emotion_id],
                    "emotion": EMOTION_NAMES[emotion_id],
                    "emotion_id": emotion_id,
                    "confidence": float(emotion_probs[emotion_id]),
                    "valence": float(emotion_dimensions[0]),
                    "arousal": float(emotion_dimensions[1]),
                    "all_probs": emotion_probs.tolist(),
                    "degraded_mode": degraded,
                }
            )
        return results

    def health(self) -> Dict[str, Any]:
        import os

        cuda_available = torch.cuda.is_available()
        return {
            "loaded": self._loaded,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "config_exists": os.path.isfile(self.config_path),
            "checkpoint_exists": os.path.isfile(self.checkpoint_path),
            "device": str(self.device) if self.device else None,
            "cuda_available": cuda_available,
        }

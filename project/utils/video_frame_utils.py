"""
视频抽帧工具：训练 dataset 与在线 inference 共用，保证抽帧逻辑一致。
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)

VIDEO_MIME_PREFIXES = ("video/",)
IMAGE_JPEG_MIMES = ("image/jpeg", "image/jpg", "image/pjpeg")


def is_video_mime(mime: str) -> bool:
    m = (mime or "").lower().strip()
    return any(m.startswith(p) for p in VIDEO_MIME_PREFIXES)


def is_image_mime(mime: str) -> bool:
    m = (mime or "").lower().strip()
    return m in IMAGE_JPEG_MIMES or m.startswith("image/")


def suffix_for_mime(mime: str, filename: str = "") -> str:
    m = (mime or "").lower()
    name = (filename or "").lower()
    if "webm" in m or name.endswith(".webm"):
        return ".webm"
    if "mp4" in m or name.endswith(".mp4"):
        return ".mp4"
    if "quicktime" in m or name.endswith(".mov"):
        return ".mov"
    if "jpeg" in m or "jpg" in m or name.endswith((".jpg", ".jpeg")):
        return ".jpg"
    return ".webm"


def probe_video_duration_sec(video_bytes: bytes, suffix: str = ".webm") -> float:
    """探测 webm/mp4 时长（秒）；失败返回 0。"""
    if not video_bytes or cv2 is None:
        return 0.0
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="emotion_probe_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return 0.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0 and total > 0:
                return total / fps
            # 部分 webm FRAME_COUNT 不可靠：顺序读估时长
            n = 0
            while cap.isOpened() and n < 900:
                ret, _ = cap.read()
                if not ret:
                    break
                n += 1
            if fps <= 0 or fps > 120:
                fps = 30.0
            return n / fps if n else 0.0
        finally:
            cap.release()
    except Exception:
        return 0.0
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def frames_list_to_tensor(frames: list, num_frames: int) -> torch.Tensor:
    if not frames:
        raise ValueError("frames_list_to_tensor: empty frames")
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))

    arr = np.array(frames, dtype=np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr).unsqueeze(0)


def decode_image_bytes_to_rgb(image_bytes: bytes, frame_size: int) -> Optional[np.ndarray]:
    """解码单张 JPEG/PNG 为 RGB uint8 (H,W,3)。"""
    if not image_bytes:
        return None
    if cv2 is not None:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.resize(frame, (frame_size, frame_size))
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image.resize((frame_size, frame_size)), dtype=np.uint8)
    except Exception:
        return None


def preprocess_video_from_frame_bytes_list(
    frame_bytes_list: List[bytes],
    config: dict,
) -> Tuple[Optional[torch.Tensor], str, int]:
    """把多张 JPEG 字节均匀组成训练同构的 (1,T,C,H,W) clip。"""
    video_config = config.get("data", {}).get("video", {})
    num_frames = int(video_config.get("num_frames", 4))
    frame_size = int(video_config.get("frame_size", 112))
    frames: List[np.ndarray] = []
    for raw in frame_bytes_list or []:
        rgb = decode_image_bytes_to_rgb(raw, frame_size)
        if rgb is not None:
            frames.append(rgb)
    if not frames:
        return None, "empty", 0
    extracted = len(frames)
    tensor = frames_list_to_tensor(frames, num_frames)
    mode = "multi_frame_sequence" if extracted > 1 else "single_frame_fallback"
    return tensor, mode, extracted


def sample_frames_from_capture(
    cap: "cv2.VideoCapture",
    num_frames: int,
    frame_size: int,
    clip_duration_sec: Optional[float] = None,
    clip_offset_sec: float = 0.0,
    *,
    sample_full_clip: bool = False,
) -> Tuple[List[np.ndarray], int]:
    """与 MultimodalDataset._load_video 一致的抽帧策略。

    - sample_full_clip=True：全片均匀 linspace（与训练一致，适合 agent 短 clip）
    - clip_offset_sec>0：从起点取 duration 窗口（时序多窗）
    - 仅 duration 且 offset=0：旧逻辑取尾部窗口（兼容）
    """
    frames: List[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0 or fps > 120:
        fps = 30.0

    def _append_frame(frame_bgr) -> None:
        if frame_bgr is None:
            return
        frame_bgr = cv2.resize(frame_bgr, (frame_size, frame_size))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    if total_frames > 0:
        start_frame = 0
        end_frame = max(total_frames - 1, 0)
        if not sample_full_clip and clip_duration_sec and clip_duration_sec > 0:
            clip_frames = max(int(clip_duration_sec * fps), num_frames)
            offset_frames = max(0, int(clip_offset_sec * fps))
            if clip_offset_sec > 0:
                start_frame = min(offset_frames, max(0, total_frames - 1))
                end_frame = min(start_frame + clip_frames - 1, total_frames - 1)
            else:
                start_frame = max(0, total_frames - clip_frames)
        indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret:
                _append_frame(frame)
    else:
        # 某些容器 FRAME_COUNT 不可靠：顺序读尽量多帧再窗口化
        max_read = max(num_frames * 90, 300)
        read_count = 0
        buffer: List[np.ndarray] = []
        while cap.isOpened() and read_count < max_read:
            ret, frame = cap.read()
            read_count += 1
            if not ret:
                break
            frame_bgr = cv2.resize(frame, (frame_size, frame_size))
            buffer.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not sample_full_clip and clip_duration_sec and clip_duration_sec > 0 and buffer:
            keep = max(int(clip_duration_sec * fps), num_frames)
            if clip_offset_sec > 0:
                skip = max(0, int(clip_offset_sec * fps))
                buffer = buffer[skip : skip + keep]
            else:
                buffer = buffer[-keep:]
        frames = buffer

    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < num_frames and len(frames) > 0:
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    return frames, len(frames)


def preprocess_video_from_file_path(
    video_path: str,
    config: dict,
    clip_offset_sec: float = 0.0,
    clip_duration_sec: Optional[float] = None,
    *,
    sample_full_clip: bool = False,
) -> Optional[torch.Tensor]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video preprocessing")
    video_config = config.get("data", {}).get("video", {})
    frame_size = video_config.get("frame_size", 112)
    num_frames = video_config.get("num_frames", 4)
    if sample_full_clip:
        clip_duration_sec = None
    elif clip_duration_sec is None:
        clip_duration_sec = config.get("data", {}).get("audio", {}).get("duration", 3.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", video_path)
        return None
    try:
        frames, extracted = sample_frames_from_capture(
            cap,
            num_frames,
            frame_size,
            clip_duration_sec=clip_duration_sec,
            clip_offset_sec=clip_offset_sec,
            sample_full_clip=sample_full_clip,
        )
    finally:
        cap.release()

    if not frames:
        return None
    logger.debug(
        "video file %s extracted %d frames full_clip=%s",
        video_path,
        extracted,
        sample_full_clip,
    )
    return frames_list_to_tensor(frames, num_frames)


def preprocess_video_window_from_file_path(
    video_path: str,
    config: dict,
    offset_sec: float,
    duration_sec: float,
) -> Optional[torch.Tensor]:
    return preprocess_video_from_file_path(
        video_path,
        config,
        clip_offset_sec=offset_sec,
        clip_duration_sec=duration_sec,
    )


def preprocess_video_from_file_bytes(
    video_bytes: bytes,
    config: dict,
    suffix: str = ".webm",
    clip_offset_sec: float = 0.0,
    clip_duration_sec: Optional[float] = None,
    *,
    sample_full_clip: bool = False,
) -> Optional[torch.Tensor]:
    if not video_bytes:
        return None
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video preprocessing")

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="emotion_video_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)
        return preprocess_video_from_file_path(
            tmp_path,
            config,
            clip_offset_sec=clip_offset_sec,
            clip_duration_sec=clip_duration_sec,
            sample_full_clip=sample_full_clip,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def preprocess_video_window_from_bytes(
    video_bytes: bytes,
    config: dict,
    offset_sec: float,
    duration_sec: float,
    suffix: str = ".webm",
) -> Optional[torch.Tensor]:
    return preprocess_video_from_file_bytes(
        video_bytes,
        config,
        suffix=suffix,
        clip_offset_sec=offset_sec,
        clip_duration_sec=duration_sec,
    )

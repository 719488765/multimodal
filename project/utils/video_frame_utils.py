"""
视频抽帧工具：训练 dataset 与在线 inference 共用，保证抽帧逻辑一致。
"""

from __future__ import annotations

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


def sample_frames_from_capture(
    cap: "cv2.VideoCapture",
    num_frames: int,
    frame_size: int,
    clip_duration_sec: Optional[float] = None,
    clip_offset_sec: float = 0.0,
) -> Tuple[List[np.ndarray], int]:
    """与 MultimodalDataset._load_video 一致的抽帧策略。

    clip_duration_sec 非空时取指定时长窗口；clip_offset_sec 为窗口起点（秒）。
    clip_offset_sec=0 且未指定 duration 时取全片；仅 duration 时取尾部窗口（兼容旧逻辑）。
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
        if clip_duration_sec and clip_duration_sec > 0:
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
        max_read = max(num_frames * 12, num_frames)
        read_count = 0
        buffer: List[np.ndarray] = []
        while cap.isOpened() and read_count < max_read:
            ret, frame = cap.read()
            read_count += 1
            if not ret:
                break
            frame_bgr = cv2.resize(frame, (frame_size, frame_size))
            buffer.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if clip_duration_sec and clip_duration_sec > 0 and buffer:
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
) -> Optional[torch.Tensor]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video preprocessing")
    video_config = config.get("data", {}).get("video", {})
    frame_size = video_config.get("frame_size", 112)
    num_frames = video_config.get("num_frames", 4)
    if clip_duration_sec is None:
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
        )
    finally:
        cap.release()

    if not frames:
        return None
    logger.debug("video file %s extracted %d frames", video_path, extracted)
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

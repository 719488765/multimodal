from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

_TTL_SEC = 600
_buffers: Dict[str, "SessionUpload"] = {}


@dataclass
class SessionUpload:
    audio_parts: Dict[int, bytes] = field(default_factory=dict)
    video_parts: Dict[int, bytes] = field(default_factory=dict)
    audio_total: int = 0
    video_total: int = 0
    updated: float = field(default_factory=time.time)


def _gc() -> None:
    now = time.time()
    stale = [k for k, v in _buffers.items() if now - v.updated > _TTL_SEC]
    for k in stale:
        _buffers.pop(k, None)


def put_chunk(session_id: str, field: str, chunk_index: int, total_chunks: int, data: bytes) -> None:
    _gc()
    buf = _buffers.setdefault(session_id, SessionUpload())
    buf.updated = time.time()
    if field == "audio":
        buf.audio_total = total_chunks
        buf.audio_parts[chunk_index] = data
    elif field == "video":
        buf.video_total = total_chunks
        buf.video_parts[chunk_index] = data
    else:
        raise ValueError(f"unknown field: {field}")


def _assemble_parts(parts: Dict[int, bytes], total: int) -> bytes:
    if total <= 0:
        return b""
    if len(parts) < total:
        missing = [i for i in range(total) if i not in parts]
        raise ValueError(f"missing chunks: {missing[:5]}")
    return b"".join(parts[i] for i in range(total))


def assemble(session_id: str) -> Tuple[bytes, bytes]:
    buf = _buffers.get(session_id)
    if buf is None:
        raise ValueError(f"no upload buffer for session {session_id}")
    audio = _assemble_parts(buf.audio_parts, buf.audio_total)
    video = _assemble_parts(buf.video_parts, buf.video_total)
    _buffers.pop(session_id, None)
    return audio, video

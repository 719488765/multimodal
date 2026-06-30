from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List


@dataclass
class Chunk:
    client_ts: float
    video_chunk_b64: str
    audio_chunk_b64: str


class IngestBuffer:
    """
    Keep per-session media chunks and expose sliding windows.

    This is an in-memory MVP implementation for quick iteration.
    """

    def __init__(self, window_size_sec: float = 3.0, step_sec: float = 1.0) -> None:
        self.window_size_sec = window_size_sec
        self.step_sec = step_sec
        self._chunks: Dict[str, Deque[Chunk]] = defaultdict(deque)
        self._last_window_ts: Dict[str, float] = defaultdict(float)

    def push_chunk(self, session_id: str, client_ts: float, video_chunk_b64: str, audio_chunk_b64: str) -> None:
        self._chunks[session_id].append(
            Chunk(client_ts=client_ts or time.time(), video_chunk_b64=video_chunk_b64, audio_chunk_b64=audio_chunk_b64)
        )
        # prevent unbounded growth in MVP.
        while len(self._chunks[session_id]) > 120:
            self._chunks[session_id].popleft()

    def build_windows(self, session_id: str) -> List[dict]:
        now = time.time()
        last = self._last_window_ts[session_id]
        if now - last < self.step_sec:
            return []
        self._last_window_ts[session_id] = now

        chunks = list(self._chunks[session_id])
        if not chunks:
            return []

        cutoff = now - self.window_size_sec
        selected = [c for c in chunks if c.client_ts >= cutoff]
        if not selected:
            selected = chunks[-1:]

        return [
            {
                "session_id": session_id,
                "window_end_ts": now,
                "video_chunk_b64": selected[-1].video_chunk_b64,
                "audio_chunk_b64": selected[-1].audio_chunk_b64,
                "chunk_count": len(selected),
            }
        ]

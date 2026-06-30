"""从 CMU-MOSEI SDK TimestampedWords.csd 提取 segment 级文本。"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np

# 现有占位符格式（organize 脚本写入）
_PLACEHOLDER_RE = re.compile(
    r"^(?:Placeholder transcript|Transcript) for (?P<vid>.+?) segment (?P<seg>\d+)\s*$",
    re.MULTILINE,
)


def is_mosei_placeholder_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return bool(_PLACEHOLDER_RE.match(t))


def parse_mosei_placeholder(text: str) -> Optional[Tuple[str, int]]:
    m = _PLACEHOLDER_RE.match((text or "").strip())
    if not m:
        return None
    return m.group("vid"), int(m.group("seg"))


def _decode_word_token(token) -> str:
    if isinstance(token, np.ndarray):
        token = token[0] if token.size else b""
    if isinstance(token, bytes):
        return token.decode("utf-8", errors="ignore").strip()
    return str(token).strip()


def extract_segment_text(
    words_entry,
    labels_entry,
    segment_idx: int,
    *,
    drop_sp: bool = True,
) -> str:
    """
    按 label segment 时间区间，从 TimestampedWords 聚合词序列。

    Args:
        words_entry: words[video_id] dict with intervals/features
        labels_entry: labels[video_id] dict with intervals
        segment_idx: segment index in labels
        drop_sp: 是否去掉 MOSEI 中的 'sp' 分词标记
    """
    intervals = labels_entry["intervals"]
    if segment_idx < 0 or segment_idx >= len(intervals):
        return ""

    seg_start, seg_end = float(intervals[segment_idx][0]), float(intervals[segment_idx][1])
    word_intervals = words_entry["intervals"]
    word_features = words_entry["features"]

    tokens = []
    for i in range(len(word_intervals)):
        ws, we = float(word_intervals[i][0]), float(word_intervals[i][1])
        mid = (ws + we) / 2.0
        if seg_start <= mid <= seg_end:
            w = _decode_word_token(word_features[i])
            if not w:
                continue
            if drop_sp and w.lower() == "sp":
                continue
            tokens.append(w)

    return " ".join(tokens).strip()

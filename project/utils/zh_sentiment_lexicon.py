"""
中文 ASR 情感词表与匹配（校准 + 仲裁共用，避免两处漂移）。
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Tuple

# 情绪 id: 0 happy, 1 sad, 2 angry, 3 fear, 4 neutral, 5 anxious, 6 other

POSITIVE_PATTERNS: List[Tuple[int, Pattern[str]]] = [
    (
        0,
        re.compile(
            r"高兴|开心|快乐|愉快|喜悦|兴奋|成功|太好了|真棒|不错|哈哈|呵呵|"
            r"好开心|很高兴|笑|yyds|绝绝子|太赞|爽|美滋滋",
            re.I,
        ),
    ),
]

NEGATIVE_PATTERNS: List[Tuple[int, Pattern[str]]] = [
    (1, re.compile(r"难过|伤心|悲伤|失落|沮丧|想哭|痛苦|不开心|难受|郁闷|好惨|委屈|破防|emo了?", re.I)),
    (2, re.compile(r"生气|愤怒|恼火|气愤|烦死了|讨厌|火大|气死|栓Q", re.I)),
    (3, re.compile(r"害怕|恐惧|担心|紧张|慌|吓|可怕|好怕", re.I)),
    (5, re.compile(r"焦虑|不安|烦躁|压力大|好烦|烦心", re.I)),
]

NEUTRAL_PATTERN = re.compile(r"还好|一般|普通|没什么|平静|冷静|正常|今天.*度|天气", re.I)

LAUGHTER_ONLY_RE = re.compile(r"^(哈{2,}|呵{2,}|[哈呵！!]+)+$", re.I)

NEGATION_HAPPY_RE = re.compile(
    r"(不|没|并非|并不|不算|不太|没有).{0,6}(高兴|开心|快乐|愉快|兴奋|成功|棒)",
    re.I,
)
NEGATION_SAD_RE = re.compile(
    r"(不|没|并非|并不).{0,6}(难过|伤心|悲伤|沮丧|难受|郁闷)",
    re.I,
)

INTENSITY_RE = re.compile(r"(特别|非常|超级|太|极其|好|真的|实在)", re.I)

EN_POSITIVE_RE = re.compile(r"\b(happy|glad|joy|great|wonderful|laugh|haha|lol)\b", re.I)
EN_NEGATIVE_SAD_RE = re.compile(r"\b(sad|unhappy|depressed|miserable|crying)\b", re.I)
EN_NEGATIVE_ANGRY_RE = re.compile(r"\b(angry|mad|furious|annoyed)\b", re.I)
EN_NEGATIVE_FEAR_RE = re.compile(r"\b(scared|afraid|fear|worried|nervous)\b", re.I)


def detect_language(text: str, default: str = "zh") -> str:
    t = (text or "").strip()
    if not t:
        return "unknown"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
    latin = len(re.findall(r"[a-zA-Z]", t))
    if cjk >= 2 and latin >= 4:
        return "mixed"
    if cjk >= 1:
        return "zh"
    if latin >= 3:
        return "en"
    return default


def has_negated_happy(text: str) -> bool:
    return bool(NEGATION_HAPPY_RE.search(text or ""))


def has_negated_sad(text: str) -> bool:
    return bool(NEGATION_SAD_RE.search(text or ""))


def intensity_factor(text: str) -> float:
    return 1.25 if INTENSITY_RE.search(text or "") else 1.0


def match_zh_sentiment(text: str) -> Optional[int]:
    """从中文/混合 ASR 推断目标情绪 id；含否定检测。"""
    if not text or not text.strip():
        return None
    t = text.strip()
    if LAUGHTER_ONLY_RE.match(t):
        if has_negated_happy(t):
            return None
        return 0
    if has_negated_happy(t):
        for eid, pat in NEGATIVE_PATTERNS:
            if pat.search(t):
                return eid
        return 4
    if has_negated_sad(t):
        return 4
    for eid, pat in POSITIVE_PATTERNS:
        if pat.search(t):
            return eid
    for eid, pat in NEGATIVE_PATTERNS:
        if pat.search(t):
            return eid
    if NEUTRAL_PATTERN.search(t):
        return 4
    if EN_POSITIVE_RE.search(t):
        return 0
    if EN_NEGATIVE_SAD_RE.search(t):
        return 1
    if EN_NEGATIVE_ANGRY_RE.search(t):
        return 2
    if EN_NEGATIVE_FEAR_RE.search(t):
        return 3
    return None

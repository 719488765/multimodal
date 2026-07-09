#!/usr/bin/env python3
"""Unit tests for shared Chinese sentiment lexicon."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.zh_sentiment_lexicon import (  # noqa: E402
    detect_language,
    has_negated_happy,
    match_zh_sentiment,
)


def test_detect_language():
    assert detect_language("我很高兴") == "zh"
    assert detect_language("hello world") == "en"
    assert detect_language("") == "unknown"


def test_match_happy():
    assert match_zh_sentiment("哈哈哈哈") == 0
    assert match_zh_sentiment("我很难过") == 1


def test_negation():
    assert has_negated_happy("我并不是很开心")
    assert match_zh_sentiment("我并不是很开心") == 4


if __name__ == "__main__":
    test_detect_language()
    test_match_happy()
    test_negation()
    print("OK")

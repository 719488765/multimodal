#!/usr/bin/env python3
"""从 CMU-MOSEI SDK TimestampedWords.csd 补全 data/*/text/mosei_*.txt。"""

from __future__ import annotations

import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.mosei_text_utils import (  # noqa: E402
    extract_segment_text,
    is_mosei_placeholder_text,
    parse_mosei_placeholder,
)

SPLITS = ("train", "val", "test")
DEFAULT_RAW = os.path.join(PROJECT_ROOT, "downloads", "CMU_MOSEI_raw", "CMU-MOSEI")


def load_sdk(raw_dir: str):
    from mmsdk import mmdatasdk

    words_path = os.path.join(raw_dir, "languages", "CMU_MOSEI_TimestampedWords.csd")
    labels_path = os.path.join(raw_dir, "labels", "CMU_MOSEI_Labels.csd")
    if not os.path.isfile(words_path):
        raise FileNotFoundError(f"未找到: {words_path}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"未找到: {labels_path}")

    print(f"加载 words:  {words_path}")
    print(f"加载 labels: {labels_path}")
    words = mmdatasdk.computational_sequence(words_path)
    labels = mmdatasdk.computational_sequence(labels_path)
    return words, labels


def count_mosei_text(data_root: str, splits: tuple[str, ...]) -> tuple[int, int, int]:
    total = placeholder = real = 0
    for split in splits:
        text_dir = os.path.join(data_root, split, "text")
        if not os.path.isdir(text_dir):
            continue
        for name in os.listdir(text_dir):
            if not name.startswith("mosei_") or not name.endswith(".txt"):
                continue
            total += 1
            with open(os.path.join(text_dir, name), encoding="utf-8") as f:
                content = f.read()
            if is_mosei_placeholder_text(content):
                placeholder += 1
            else:
                real += 1
    return total, placeholder, real


def collect_jobs(data_root: str, splits: tuple[str, ...], overwrite: bool) -> list[str]:
    jobs = []
    for split in splits:
        text_dir = os.path.join(data_root, split, "text")
        if not os.path.isdir(text_dir):
            continue
        for name in sorted(os.listdir(text_dir)):
            if not name.startswith("mosei_") or not name.endswith(".txt"):
                continue
            path = os.path.join(text_dir, name)
            if not overwrite:
                with open(path, encoding="utf-8") as f:
                    if not is_mosei_placeholder_text(f.read()):
                        continue
            jobs.append(path)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MOSEI text from SDK TimestampedWords.csd")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有非占位文本")
    parser.add_argument("--keep-sp", action="store_true", help="保留 MOSEI 词表中的 sp 标记")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    data_root = os.path.abspath(args.data_root)

    total, ph, real = count_mosei_text(data_root, splits)
    jobs = collect_jobs(data_root, splits, args.overwrite)
    if args.limit > 0:
        jobs = jobs[: args.limit]

    print("==> MOSEI SDK 文本补全")
    print(f"    data_root: {data_root}")
    print(f"    raw_dir:   {args.raw_dir}")
    print(f"    mosei 文本: 总计 {total}  占位 {ph}  已有真实 {real}")
    print(f"    待处理:    {len(jobs)}")

    if args.dry_run:
        for p in jobs[:5]:
            with open(p, encoding="utf-8") as f:
                parsed = parse_mosei_placeholder(f.read())
            print(f"    [dry-run] {p} -> {parsed}")
        if len(jobs) > 5:
            print(f"    ... 共 {len(jobs)} 条")
        return 0

    if not jobs:
        print("    无需处理。")
        return 0

    words, labels = load_sdk(args.raw_dir)

    ok = empty = fail = 0
    failures: list[str] = []

    for i, text_path in enumerate(jobs, start=1):
        try:
            with open(text_path, encoding="utf-8") as f:
                content = f.read()
            parsed = parse_mosei_placeholder(content)
            if parsed is None:
                fail += 1
                failures.append(f"{text_path}: cannot parse placeholder")
                continue

            vid, seg_idx = parsed
            if vid not in words.data or vid not in labels.data:
                fail += 1
                failures.append(f"{text_path}: video {vid} not in SDK")
                continue

            text = extract_segment_text(
                words[vid],
                labels[vid],
                seg_idx,
                drop_sp=not args.keep_sp,
            )
            if not text:
                empty += 1
                failures.append(f"{text_path}: empty transcript vid={vid} seg={seg_idx}")

            with open(text_path, "w", encoding="utf-8") as f:
                f.write((text or content.strip()) + "\n")
            ok += 1
        except Exception as exc:
            fail += 1
            failures.append(f"{text_path}: {exc}")

        if i % 1000 == 0 or i == len(jobs):
            print(f"    进度 {i}/{len(jobs)} ok={ok} empty={empty} fail={fail}")

    total2, ph2, real2 = count_mosei_text(data_root, splits)
    print("")
    print("==> 完成")
    print(f"    写入: {ok}  空文本: {empty}  失败: {fail}")
    print(f"    文本统计: 总计 {total2}  占位 {ph2}  真实 {real2}")

    if failures:
        log_path = os.path.join(PROJECT_ROOT, "logs_accuracy_seq", "mosei_text_extract_failures.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))
        print(f"    失败/空文本列表: {log_path}")

    return 2 if fail > 0 and ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

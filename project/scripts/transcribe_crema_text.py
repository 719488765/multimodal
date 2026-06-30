#!/usr/bin/env python3
"""用 ASR 为 CREMA 样本生成真实 text/*.txt（替换占位符）。"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.asr_utils import (  # noqa: E402
    is_crema_placeholder_text,
    normalize_transcript,
    transcribe_audio_file,
)

SPLITS = ("train", "val", "test")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")


def resolve_crema_audio(audio_dir: str, sample_id: str) -> str | None:
    for ext in AUDIO_EXTS:
        path = os.path.join(audio_dir, f"{sample_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


def count_crema_text(data_root: str, splits: tuple[str, ...]) -> tuple[int, int, int]:
    total = placeholder = real = 0
    for split in splits:
        text_dir = os.path.join(data_root, split, "text")
        if not os.path.isdir(text_dir):
            continue
        for name in os.listdir(text_dir):
            if not name.startswith("crema_") or not name.endswith(".txt"):
                continue
            total += 1
            with open(os.path.join(text_dir, name), encoding="utf-8") as f:
                content = f.read()
            if is_crema_placeholder_text(content):
                placeholder += 1
            else:
                real += 1
    return total, placeholder, real


def collect_jobs(data_root: str, splits: tuple[str, ...], overwrite: bool) -> list[tuple[str, str, str]]:
    jobs = []
    for split in splits:
        audio_dir = os.path.join(data_root, split, "audio")
        text_dir = os.path.join(data_root, split, "text")
        if not os.path.isdir(audio_dir):
            continue
        os.makedirs(text_dir, exist_ok=True)
        for name in sorted(os.listdir(audio_dir)):
            if not name.startswith("crema_"):
                continue
            base, _ = os.path.splitext(name)
            text_path = os.path.join(text_dir, f"{base}.txt")
            if not overwrite and os.path.isfile(text_path):
                with open(text_path, encoding="utf-8") as f:
                    if not is_crema_placeholder_text(f.read()):
                        continue
            audio_path = os.path.join(audio_dir, name)
            jobs.append((split, audio_path, text_path))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe CREMA audio to text via ASR")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--engine", default="auto", choices=["auto", "faster_whisper", "transformers"])
    parser.add_argument("--model", default="base", help="faster-whisper size or openai/whisper-* id")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    data_root = os.path.abspath(args.data_root)

    total, ph, real = count_crema_text(data_root, splits)
    jobs = collect_jobs(data_root, splits, args.overwrite)
    if args.limit > 0:
        jobs = jobs[: args.limit]

    print("==> CREMA ASR 文本补全")
    print(f"    data_root: {data_root}")
    print(f"    splits:    {splits}")
    print(f"    crema 文本: 总计 {total}  占位 {ph}  已有真实 {real}")
    print(f"    待处理:    {len(jobs)}  engine={args.engine}  device={args.device}")

    if args.dry_run:
        for _, audio_path, text_path in jobs[:5]:
            print(f"    [dry-run] {audio_path} -> {text_path}")
        if len(jobs) > 5:
            print(f"    ... 共 {len(jobs)} 条")
        return 0

    if not jobs:
        print("    无需处理。")
        return 0

    ok = skip = fail = 0
    failures: list[str] = []

    for i, (split, audio_path, text_path) in enumerate(jobs, start=1):
        try:
            text, eng = transcribe_audio_file(
                audio_path,
                engine=args.engine,
                model=args.model,
                device=args.device,
            )
            text = normalize_transcript(text)
            if not text:
                raise RuntimeError("empty transcript")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            ok += 1
        except Exception as exc:
            fail += 1
            failures.append(f"{audio_path}: {exc}")

        if i % 100 == 0 or i == len(jobs):
            print(f"    进度 {i}/{len(jobs)} ok={ok} fail={fail}")

    total2, ph2, real2 = count_crema_text(data_root, splits)
    print("")
    print("==> 完成")
    print(f"    成功: {ok}  失败: {fail}")
    print(f"    文本统计: 总计 {total2}  占位 {ph2}  真实 {real2}")

    if failures:
        log_path = os.path.join(PROJECT_ROOT, "logs_accuracy_seq", "crema_asr_failures.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))
        print(f"    失败列表: {log_path}")
        return 2 if ok == 0 else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

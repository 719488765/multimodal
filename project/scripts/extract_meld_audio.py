#!/usr/bin/env python3
"""从已整理的 MELD mp4 批量提取 mono 16kHz WAV 到 data/{split}/audio/。"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.meld_audio_utils import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    _worker_extract,
    extract_audio_from_video,
    ffmpeg_available,
    meld_wav_path_for_video,
    pyav_available,
)

SPLITS = ("train", "val", "test")


def collect_jobs(data_root: str, splits: tuple[str, ...], overwrite: bool) -> list[tuple[str, str, int, bool]]:
    jobs: list[tuple[str, str, int, bool]] = []
    for split in splits:
        video_dir = os.path.join(data_root, split, "video")
        audio_dir = os.path.join(data_root, split, "audio")
        if not os.path.isdir(video_dir):
            continue
        os.makedirs(audio_dir, exist_ok=True)
        for name in sorted(os.listdir(video_dir)):
            if not name.startswith("meld_") or not name.lower().endswith(".mp4"):
                continue
            video_path = os.path.join(video_dir, name)
            wav_path = meld_wav_path_for_video(video_path, audio_dir)
            if not overwrite and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 44:
                continue
            jobs.append((video_path, wav_path, DEFAULT_SAMPLE_RATE, overwrite))
    return jobs


def count_meld_pairs(data_root: str, splits: tuple[str, ...]) -> tuple[int, int]:
    videos = audios = 0
    for split in splits:
        video_dir = os.path.join(data_root, split, "video")
        audio_dir = os.path.join(data_root, split, "audio")
        if not os.path.isdir(video_dir):
            continue
        for name in os.listdir(video_dir):
            if not name.startswith("meld_") or not name.lower().endswith(".mp4"):
                continue
            videos += 1
            wav = os.path.join(audio_dir, os.path.splitext(name)[0] + ".wav")
            if os.path.isfile(wav) and os.path.getsize(wav) > 44:
                audios += 1
    return videos, audios


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract MELD audio (16kHz mono WAV) from mp4")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--splits", default=",".join(SPLITS), help="train,val,test")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry-failures",
        default="",
        help="从失败列表 txt 重试（每行: video_path: reason）",
    )
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    data_root = os.path.abspath(args.data_root)

    if not pyav_available() and not ffmpeg_available():
        print("ERROR: 需要 PyAV (pip install av) 或 ffmpeg", file=sys.stderr)
        return 1

    if args.retry_failures:
        jobs = []
        with open(args.retry_failures, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_path = line.split(":", 1)[0].strip()
                if not video_path.endswith(".mp4"):
                    continue
                wav_path = meld_wav_path_for_video(video_path)
                jobs.append((video_path, wav_path, DEFAULT_SAMPLE_RATE, True))
        total_v, have_a = count_meld_pairs(data_root, splits)
    else:
        total_v, have_a = count_meld_pairs(data_root, splits)
        jobs = collect_jobs(data_root, splits, args.overwrite)

    print("==> MELD 音频提取")
    print(f"    data_root: {data_root}")
    print(f"    splits:    {splits}")
    print(f"    meld 视频: {total_v}  已有 wav: {have_a}  待提取: {len(jobs)}")
    print(f"    workers:   {args.workers}  sample_rate: {DEFAULT_SAMPLE_RATE} Hz")

    if args.dry_run:
        for v, w, _, _ in jobs[:5]:
            print(f"    [dry-run] {v} -> {w}")
        if len(jobs) > 5:
            print(f"    ... 共 {len(jobs)} 条")
        return 0

    if not jobs:
        print("    无需提取（已全部存在）。可直接训练。")
        return 0

    ok_n = skip_n = fail_n = 0
    failures: list[str] = []

    if args.workers <= 1:
        for job in jobs:
            video_path, wav_path, sr, ow = job
            ok, msg = extract_audio_from_video(video_path, wav_path, sample_rate=sr, overwrite=ow)
            if ok and msg == "skipped existing":
                skip_n += 1
            elif ok:
                ok_n += 1
            else:
                fail_n += 1
                failures.append(f"{video_path}: {msg}")
            done = ok_n + skip_n + fail_n
            if done % 500 == 0:
                print(f"    进度 {done}/{len(jobs)} (ok={ok_n} fail={fail_n})")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker_extract, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), start=1):
                video_path, ok, msg = fut.result()
                if ok and msg == "skipped existing":
                    skip_n += 1
                elif ok:
                    ok_n += 1
                else:
                    fail_n += 1
                    failures.append(f"{video_path}: {msg}")
                if i % 500 == 0:
                    print(f"    进度 {i}/{len(jobs)} (ok={ok_n} fail={fail_n})")

    total_v2, have_a2 = count_meld_pairs(data_root, splits)
    print("")
    print("==> 完成")
    print(f"    新提取: {ok_n}  跳过: {skip_n}  失败: {fail_n}")
    print(f"    meld 视频: {total_v2}  已有 wav: {have_a2}")

    if failures:
        log_path = os.path.join(PROJECT_ROOT, "logs_accuracy_seq", "meld_audio_extract_failures.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))
        print(f"    失败列表: {log_path} ({len(failures)} 条)")
        return 2 if ok_n == 0 else 0

    if have_a2 < total_v2:
        print(f"    WARN: 仍有 {total_v2 - have_a2} 个视频无对应 wav")
        return 2

    print("    全部 MELD 样本已有音频，可以开始 train_meld_agent.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""从 zh_cases benchmark 生成 agent_capture 训练样本（文本+标签，复用 MELD A/V）。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "agent_benchmark" / "zh_cases.json"
CAPTURE_ROOT = ROOT / "data" / "agent_capture"
MELD_AUDIO = ROOT / "data" / "train" / "audio"
MELD_VIDEO = ROOT / "data" / "train" / "video"

EMOTION_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "neutral": "neutral",
    "anxious": "fear",
    "other": "neutral",
    "surprise": "surprise",
    "disgust": "disgust",
}


def _pick_meld_av(index: int) -> tuple[Path | None, Path | None]:
    stem = f"meld_train_{index:04d}"
    audio = MELD_AUDIO / f"{stem}.wav"
    video = None
    for ext in (".mp4", ".flv", ".webm"):
        cand = MELD_VIDEO / f"{stem}{ext}"
        if cand.is_file():
            video = cand
            break
    return (audio if audio.is_file() else None, video)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, default=100, help="目标样本数（循环 benchmark 扩充）")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dst", type=Path, default=CAPTURE_ROOT)
    args = parser.parse_args()

    cases = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if not cases:
        print("[FAIL] empty zh_cases.json")
        return 1

    dst = args.dst / args.split
    for sub in ("video", "audio", "text", "labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    created = 0
    for i in range(args.target_count):
        case = cases[i % len(cases)]
        suffix = f"{i // len(cases):02d}" if i >= len(cases) else ""
        sample_id = f"acap_{case['name']}{suffix}_{i:04d}"
        label_name = EMOTION_MAP.get(case["expect_final"], "neutral")
        text = (case.get("text") or "").strip()
        if not text:
            continue

        audio_src, video_src = _pick_meld_av((i % 2000) + 1)
        if audio_src is None:
            print(f"[WARN] skip {sample_id}: no meld audio")
            continue

        shutil.copy2(audio_src, dst / "audio" / f"{sample_id}.wav")
        if video_src is not None:
            shutil.copy2(video_src, dst / "video" / f"{sample_id}{video_src.suffix}")
        (dst / "text" / f"{sample_id}.txt").write_text(text + "\n", encoding="utf-8")
        (dst / "labels" / f"{sample_id}.txt").write_text(
            f"{label_name}\n0.0,0.0\n", encoding="utf-8"
        )
        created += 1

    manifest = {
        "source": str(BENCHMARK),
        "split": args.split,
        "target_count": args.target_count,
        "created": created,
        "note": "Reuse MELD A/V with Chinese text from zh_cases benchmark",
    }
    out_manifest = args.dst / "seed_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if created >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""将原始录制整理为 data/agent_capture 布局（占位，需按实际 raw 目录扩展）。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "data" / "agent_capture"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="原始 wav/mp4/txt 目录")
    p.add_argument("--dst", default=str(DEFAULT_DST))
    p.add_argument("--split", default="train")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) / args.split
    for sub in ("video", "audio", "text", "labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    n = 0
    for wav in sorted(src.glob("*.wav")):
        stem = wav.stem
        vid = next(src.glob(f"{stem}.*"), None)
        label = src / f"{stem}.label.txt"
        if not label.is_file():
            label = src / f"{stem}.txt"
        (dst / "audio" / f"meld_{args.split}_{stem}.wav").write_bytes(wav.read_bytes())
        if label.is_file():
            txt = label.read_text(encoding="utf-8").strip().splitlines()[0]
            (dst / "labels" / f"meld_{args.split}_{stem}.txt").write_text(
                txt + "\n0.0,0.0\n", encoding="utf-8"
            )
        for ext in (".mp4", ".webm", ".jpg"):
            cand = src / f"{stem}{ext}"
            if cand.is_file():
                shutil.copy2(cand, dst / "video" / f"meld_{args.split}_{stem}{ext}")
                break
        n += 1

    print(f"Organized {n} samples -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

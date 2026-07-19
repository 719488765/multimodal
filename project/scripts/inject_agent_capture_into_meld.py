#!/usr/bin/env python3
"""将 agent_capture 样本注入 data/train，作为 MELD 扩展样本参与训练。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_TRAIN = ROOT / "data" / "agent_capture" / "train"
TRAIN_ROOT = ROOT / "data" / "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=CAPTURE_TRAIN)
    parser.add_argument("--prefix", default="meld_train_acap")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"[FAIL] missing capture dir: {args.src}")
        return 1

    injected = 0
    for label_path in sorted((args.src / "labels").glob("*.txt")):
        stem = label_path.stem
        out_stem = f"{args.prefix}_{stem}"
        label_text = label_path.read_text(encoding="utf-8")
        (TRAIN_ROOT / "labels" / f"{out_stem}.txt").write_text(label_text, encoding="utf-8")

        for sub, exts in (
            ("text", (".txt",)),
            ("audio", (".wav",)),
            ("video", (".mp4", ".webm", ".flv")),
        ):
            src_dir = args.src / sub
            if not src_dir.is_dir():
                continue
            for ext in exts:
                src = src_dir / f"{stem}{ext}"
                if src.is_file():
                    shutil.copy2(src, TRAIN_ROOT / sub / f"{out_stem}{ext}")
                    break
        injected += 1

    manifest = {
        "src": str(args.src),
        "prefix": args.prefix,
        "injected": injected,
        "train_labels_dir": str(TRAIN_ROOT / "labels"),
    }
    out = ROOT / "data" / "agent_benchmark" / "agent_capture_inject_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if injected > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

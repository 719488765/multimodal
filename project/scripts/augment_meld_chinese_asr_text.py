#!/usr/bin/env python3
"""
为 MELD 训练集生成中文 ASR 伪标签文本（可选数据增强）。

用法:
  python scripts/augment_meld_chinese_asr_text.py --dry-run --limit 10
  python scripts/augment_meld_chinese_asr_text.py --output-suffix _zh.txt

需 asr-local :9010 或本地 Whisper；默认仅扫描 data/meld 文本目录结构。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "agent_benchmark" / "meld_zh_asr_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-suffix", default="_zh.txt")
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    text_roots = [
        ROOT / "data" / "meld" / "train" / "text",
        ROOT / "data" / "train" / "text",
    ]
    sources = []
    for root in text_roots:
        if root.is_dir():
            sources.extend(sorted(root.glob("*.txt")))

    if args.limit > 0:
        sources = sources[: args.limit]

    manifest = {
        "note": "Run Whisper zh on paired audio to populate *_zh.txt; training yaml can mix en/zh.",
        "output_suffix": args.output_suffix,
        "sources": [str(p.relative_to(ROOT)) for p in sources],
        "count": len(sources),
    }

    if not args.dry_run:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote manifest -> {args.manifest_out}" if not args.dry_run else "dry-run only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""为 MELD train 文本生成中文增强 *_zh.txt（映射 zh_cases 或直写中文）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "agent_benchmark" / "zh_cases.json"
TEXT_DIR = ROOT / "data" / "train" / "text"
DEFAULT_MANIFEST = ROOT / "data" / "agent_benchmark" / "meld_zh_asr_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0=全部 meld_train 文本")
    parser.add_argument("--suffix", default="_zh.txt")
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    zh_texts = [c["text"] for c in cases if (c.get("text") or "").strip()]
    if not zh_texts:
        print("[FAIL] no zh texts in benchmark")
        return 1

    sources = sorted(TEXT_DIR.glob("meld_train_*.txt"))
    sources = [p for p in sources if not p.name.endswith(args.suffix)]
    if args.limit > 0:
        sources = sources[: args.limit]

    entries = []
    written = 0
    for i, src in enumerate(sources):
        zh = zh_texts[i % len(zh_texts)]
        out = src.with_name(src.stem + args.suffix)
        entries.append(
            {
                "en": str(src.relative_to(ROOT)),
                "zh": str(out.relative_to(ROOT)),
                "zh_preview": zh[:40],
            }
        )
        if not args.dry_run:
            out.write_text(zh + "\n", encoding="utf-8")
            written += 1

    manifest = {
        "note": "Chinese text augmentation for MELD train; training may mix en/zh via zh_text_mix_prob",
        "output_suffix": args.suffix,
        "sources_count": len(sources),
        "written": written,
        "entries": entries[:20],
        "entries_truncated": len(entries) > 20,
        "total_entries": len(entries),
    }

    if not args.dry_run:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"Wrote manifest -> {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""统计 train/val/test 标签分布（按数据集；支持 native / unified）。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import MultimodalDataset
from utils import load_config
from utils.label_mapping import (
    NATIVE_EMOTION_NAMES,
    UNIFIED_EMOTION_NAMES,
    get_emotion_class_names,
    uses_native_labels,
)

EMOTION_CN = ["开心", "难过", "生气", "害怕", "平静", "焦虑", "其他"]


def _name_for_id(dataset_name: str, eid: int, cfg: dict, native: bool) -> str:
    if native and dataset_name:
        names = get_emotion_class_names(dataset_name, cfg.get("datasets", {}))
        if eid < len(names):
            return names[eid]
    if eid < len(UNIFIED_EMOTION_NAMES):
        return UNIFIED_EMOTION_NAMES[eid]
    return "?"


def audit(split: str, config_path: str, native: bool) -> None:
    cfg = load_config(config_path)
    ds = MultimodalDataset(cfg["data"]["root_dir"], split=split, config=cfg)
    by_ds: dict = {}
    total = Counter()
    for idx in range(len(ds)):
        sample = ds.data_list[idx]
        eid, _ = ds._load_label(sample.get("label_path"), sample=sample)
        if eid is None:
            eid = 4
        ds_name = ds._infer_sample_dataset_name(sample) or "unknown"
        by_ds.setdefault(ds_name, Counter())[eid] += 1
        total[eid] += 1

    mode = "native" if native else "unified"
    print(f"\n=== split={split} mode={mode} n={len(ds)} ===")
    for name, cnt in sorted(by_ds.items()):
        use_native = native or uses_native_labels(name, cfg.get("datasets", {}))
        print(f"  [{name}] total={sum(cnt.values())}")
        for eid in sorted(cnt.keys()):
            label = _name_for_id(name, eid, cfg, use_native)
            cn = EMOTION_CN[eid] if not use_native and eid < 7 else label
            print(f"    id={eid} {label} ({cn}): {cnt[eid]}")
    print(f"  [ALL] total={sum(total.values())}")
    for eid in sorted(total.keys()):
        print(f"    id={eid}: {total[eid]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/sdavt_v3/meld/S1_M0_AVT_ES_baseline.yaml")
    p.add_argument(
        "--split",
        default="train,val",
        help="train,val,test 或 all",
    )
    p.add_argument(
        "--native",
        action="store_true",
        help="按 datasets.<name>.use_native_labels 或全局 native 口径统计",
    )
    args = p.parse_args()

    if args.split.strip().lower() == "all":
        splits = ("train", "val", "test")
    else:
        splits = tuple(s.strip() for s in args.split.split(",") if s.strip())

    for split in splits:
        audit(split, args.config, args.native)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

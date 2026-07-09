#!/usr/bin/env python3
"""Validate P4 modality-ablation job metrics for collapse (ln(K) uniform prediction)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"

# MOSEI 7-class ln(7)/7 ≈ 0.0881; CREMA 6-class macro-F1 collapse often < 0.06 at ep0
COLLAPSE_F1 = {
    "mosei": 0.10,
    "crema": 0.06,
    "meld": 0.10,
}
MOSEI_EP0_MIN = 0.15


def read_ep0_val_f1(metrics_csv: Path) -> float | None:
    with metrics_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") == "val" and row.get("epoch") == "0":
                f1 = row.get("f1", "").strip()
                return float(f1) if f1 else None
    return None


def read_best_val_f1(metrics_csv: Path) -> float | None:
    best = None
    with metrics_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "val":
                continue
            f1 = row.get("f1", "").strip()
            if not f1:
                continue
            v = float(f1)
            best = v if best is None else max(best, v)
    return best


def validate_job(job_id: str, dataset: str, run_dir: str | None = None) -> int:
    run_dir = run_dir or f"SDAVT_R4_{job_id}"
    metrics_csv = LOG_DIR / run_dir / "metrics.csv"
    if not metrics_csv.is_file():
        print(f"[FAIL] missing metrics: {metrics_csv}")
        return 1

    ep0 = read_ep0_val_f1(metrics_csv)
    best = read_best_val_f1(metrics_csv)
    threshold = COLLAPSE_F1[dataset]

    print(f"job={job_id} dataset={dataset} ep0_val_f1={ep0} best_val_f1={best} threshold={threshold}")

    if ep0 is None:
        print("[FAIL] no ep0 val row")
        return 1
    if dataset == "mosei" and ep0 < MOSEI_EP0_MIN:
        print(f"[FAIL] MOSEI ep0 val F1 {ep0:.4f} < {MOSEI_EP0_MIN} (ln(7) collapse risk)")
        return 1
    if best is not None and best <= threshold:
        print(f"[FAIL] best val F1 {best:.4f} <= collapse threshold {threshold}")
        return 1

    print("[OK] non-collapse")
    return 0


def validate_dataset(dataset: str) -> int:
    if not QUEUE_FILE.is_file():
        print(f"[FAIL] missing queue: {QUEUE_FILE}")
        return 1
    q = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    jobs = [
        j for j in q.get("jobs", [])
        if j.get("phase") == "p4_modal" and j.get("dataset") == dataset
    ]
    if not jobs:
        print(f"[FAIL] no p4_modal jobs for dataset={dataset}")
        return 1
    rc = 0
    for j in sorted(jobs, key=lambda x: x.get("id", "")):
        job_id = j["id"]
        run_dir = j.get("run_dir") or f"SDAVT_R4_{job_id}"
        if validate_job(job_id, dataset, run_dir) != 0:
            rc = 1
    return rc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("job_id", nargs="?", help="e.g. R4_A_C_AT (omit with --dataset --strict)")
    p.add_argument("--dataset", choices=["crema", "mosei", "meld"], required=True)
    p.add_argument("--run-dir", default=None, help="log subdir, default SDAVT_R4_{job_id}")
    p.add_argument("--strict", action="store_true", help="Validate all p4_modal jobs for dataset")
    args = p.parse_args()

    if args.strict:
        return validate_dataset(args.dataset)
    if not args.job_id:
        print("[FAIL] job_id required unless --strict")
        return 1
    return validate_job(args.job_id, args.dataset, args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())

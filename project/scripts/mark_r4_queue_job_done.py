#!/usr/bin/env python3
"""Mark a queue job done without retraining (metrics already valid)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"


def read_best(run_dir: str):
    p = LOG_DIR / run_dir / "metrics.csv"
    val_rows = []
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") == "val" and row.get("f1"):
                val_rows.append(row)
    if not val_rows:
        raise SystemExit(f"no val rows in {p}")
    best = max(val_rows, key=lambda r: float(r["f1"]))
    best_acc = max(float(r["accuracy"]) for r in val_rows if r.get("accuracy"))
    return float(best["f1"]), best_acc, int(best["epoch"]), len(val_rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("job_id")
    p.add_argument("--phase", default="p4_modal")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    f1, acc, ep, epochs = read_best(args.run_dir)
    q = json.loads(QUEUE.read_text(encoding="utf-8"))
    now = datetime.now().isoformat(timespec="seconds")
    for j in q["jobs"]:
        if j["id"] == args.job_id and j.get("phase") == args.phase:
            j["status"] = "done"
            j["run_dir"] = args.run_dir
            j["best_val_f1"] = round(f1, 6)
            j["best_val_acc"] = round(acc, 6)
            j["best_val_f1_ep"] = ep
            j["epochs_done"] = epochs
            j["updated_at"] = now
            j["note"] = "marked_done_recovery"
            break
    else:
        raise SystemExit(f"job not found: {args.job_id}")
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] {args.job_id} -> done F1={f1:.4f}@{ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

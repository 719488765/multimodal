#!/usr/bin/env python3
"""P2 ES 重训收尾：从 metrics.csv 回写队列 best 指标，导出曲线并刷新 report。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"

JOB_SLOTS = {
    "F_M_ES": "SDAVT_R4_F_M_ES",
    "F_C_ES": "SDAVT_R4_F_C_ES",
}


def read_best_metrics(run_dir: str) -> Tuple[Optional[float], Optional[float], Optional[int], int]:
    metrics = LOG_DIR / run_dir / "metrics.csv"
    if not metrics.exists():
        return None, None, None, 0
    val_rows = []
    with metrics.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "val":
                continue
            f1 = row.get("f1")
            acc = row.get("accuracy")
            if not f1:
                continue
            val_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "f1": float(f1),
                    "accuracy": float(acc) if acc else None,
                }
            )
    if not val_rows:
        return None, None, None, 0
    best = max(val_rows, key=lambda r: r["f1"])
    best_acc = max(r["accuracy"] for r in val_rows if r["accuracy"] is not None)
    return best["f1"], best_acc, best["epoch"], len(val_rows)


def update_queue(job_id: str, run_dir: str, status: str, note_suffix: str = "") -> Dict[str, Any]:
    best_f1, best_acc, best_ep, epochs = read_best_metrics(run_dir)
    q = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    now = datetime.now().isoformat(timespec="seconds")
    updated: Optional[Dict[str, Any]] = None
    for j in q["jobs"]:
        if j.get("phase") != "p2_fusion" or j["id"] != job_id:
            continue
        j["status"] = status
        j["run_dir"] = run_dir
        j["finished_at"] = now if status in ("done", "failed") else j.get("finished_at")
        j["updated_at"] = now
        if best_f1 is not None:
            j["best_val_f1"] = round(best_f1, 6)
            j["best_val_acc"] = round(best_acc, 6) if best_acc is not None else None
        note = str(j.get("note") or "")
        if note_suffix and note_suffix not in note:
            j["note"] = (note + ";" + note_suffix).strip(";")
        updated = j
        break
    if updated is None:
        raise SystemExit(f"job not found: p2_fusion/{job_id}")
    QUEUE_FILE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "job_id": job_id,
        "run_dir": run_dir,
        "status": status,
        "best_val_f1": best_f1,
        "best_val_f1_ep": best_ep,
        "best_val_acc": best_acc,
        "epochs": epochs,
    }


def refresh_artifacts(run_dir: str) -> None:
    cmds = [
        ["python3", "scripts/export_sdavt_r4_curves.py", "--run", run_dir],
        ["python3", "scripts/build_sdavt_r4_tables.py"],
        ["python3", "scripts/build_sdavt_r4_report.py"],
        ["python3", "scripts/monitor_sdavt_r4.py", "--once"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id", choices=sorted(JOB_SLOTS))
    parser.add_argument("--status", default="done", choices=("done", "failed", "running"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or JOB_SLOTS[args.job_id]
    info = update_queue(args.job_id, run_dir, args.status, args.note)
    print(
        f"[OK] queue {args.job_id} -> {args.status} "
        f"best_f1={info['best_val_f1']}@{info['best_val_f1_ep']} epochs={info['epochs']}"
    )
    if not args.no_refresh and args.status == "done" and info["epochs"] > 0:
        refresh_artifacts(run_dir)
        print(f"[OK] curves/report refreshed for {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

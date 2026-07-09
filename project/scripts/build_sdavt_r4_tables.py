#!/usr/bin/env python3
"""Build R4 summary tables including P2 fusion winners from logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
OUT_DIR = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "tables"
WINNERS_FILE = OUT_DIR / "r4_fusion_winners.json"

P2_JOBS = {
    "crema": "F_C_ES",
    "meld": "F_M_ES",
    "mosei": "F_O_ES",
}


def read_best(run_dir: str) -> Tuple[Optional[float], Optional[float], Optional[int], int, Optional[float], Optional[float]]:
    metrics = LOG_DIR / run_dir / "metrics.csv"
    if not metrics.is_file():
        return None, None, None, 0, None, None
    val_rows: List[Dict[str, Any]] = []
    with metrics.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "val" or not row.get("f1"):
                continue
            val_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "f1": float(row["f1"]),
                    "accuracy": float(row["accuracy"]) if row.get("accuracy") else None,
                    "cls_ce_uw": float(row["cls_ce_unweighted"]) if row.get("cls_ce_unweighted") else None,
                }
            )
    if not val_rows:
        return None, None, None, 0, None, None
    best_f1_row = max(val_rows, key=lambda r: r["f1"])
    best_acc_row = max(val_rows, key=lambda r: r["accuracy"] or 0.0)
    last = val_rows[-1]
    min_ce = min((r["cls_ce_uw"] for r in val_rows if r["cls_ce_uw"] is not None), default=None)
    return (
        best_f1_row["f1"],
        best_acc_row["accuracy"],
        best_f1_row["epoch"],
        len(val_rows),
        last["f1"],
        last["accuracy"],
    )


def resolve_run_dir(job_id: str, dataset: str) -> Optional[str]:
    q = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    for j in q.get("jobs", []):
        if j.get("id") == job_id and j.get("run_dir"):
            p = LOG_DIR / j["run_dir"] / "metrics.csv"
            if p.is_file():
                return j["run_dir"]
    fixed = f"SDAVT_R4_{job_id}"
    if (LOG_DIR / fixed / "metrics.csv").is_file():
        return fixed
    prefix = f"SDAVT_R4_{job_id}_"
    matches = sorted(p.parent.name for p in LOG_DIR.glob(f"{prefix}*/metrics.csv"))
    if matches:
        return matches[-1]
    if dataset == "mosei":
        matches = sorted(p.parent.name for p in LOG_DIR.glob("SDAVT_R4_F_O_ES_*/metrics.csv"))
        return matches[-1] if matches else None
    return None


def build_winners() -> Dict[str, Any]:
    winners: Dict[str, Any] = {}
    for dataset, job_id in P2_JOBS.items():
        run_dir = resolve_run_dir(job_id, dataset)
        if not run_dir:
            continue
        best_f1, best_acc, best_f1_ep, epochs, last_f1, last_acc = read_best(run_dir)
        winners[dataset] = {
            "run": run_dir,
            "phase": "p2_fusion",
            "epochs": epochs,
            "best_acc": round(best_acc, 6) if best_acc is not None else None,
            "best_acc_ep": None,
            "best_f1": round(best_f1, 6) if best_f1 is not None else None,
            "best_f1_ep": best_f1_ep,
            "last_acc": round(last_acc, 6) if last_acc is not None else None,
            "last_f1": round(last_f1, 6) if last_f1 is not None else None,
            "min_cls_ce_uw": None,
            "dataset": dataset,
            "job_id": job_id,
        }
    return winners


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    winners = build_winners()
    WINNERS_FILE.write_text(json.dumps(winners, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] fusion winners -> {WINNERS_FILE}")
    for ds, w in winners.items():
        print(f"  {ds}: F1={w.get('best_f1')} Acc={w.get('best_acc')} run={w.get('run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

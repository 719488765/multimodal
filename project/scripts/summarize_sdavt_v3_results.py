#!/usr/bin/env python3
"""汇总 logs_sdavt_v3 各 run 的 best/last 指标到 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def to_float(val: str) -> Optional[float]:
    if val is None:
        return None
    val = val.strip()
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def pick_best(rows: List[Dict], key: str) -> Dict:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return {"epoch": None, key: None}
    return max(valid, key=lambda r: r[key])


def summarize_run(metrics_csv: Path) -> Optional[Dict]:
    val_rows: List[Dict] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase", "") != "val":
                continue
            val_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": to_float(row.get("loss", "")),
                    "accuracy": to_float(row.get("accuracy", "")),
                    "precision": to_float(row.get("precision", "")),
                    "recall": to_float(row.get("recall", "")),
                    "f1": to_float(row.get("f1", "")),
                }
            )

    if not val_rows:
        return None

    val_rows.sort(key=lambda x: x["epoch"])
    last_val = val_rows[-1]
    best_acc = pick_best(val_rows, "accuracy")
    best_f1 = pick_best(val_rows, "f1")

    return {
        "run": metrics_csv.parent.name,
        "val_epochs": len(val_rows),
        "last_val_epoch": last_val["epoch"],
        "last_val_accuracy": last_val["accuracy"],
        "last_val_f1": last_val["f1"],
        "best_val_accuracy": best_acc["accuracy"],
        "best_val_accuracy_epoch": best_acc["epoch"],
        "best_val_f1": best_f1["f1"],
        "best_val_f1_epoch": best_f1["epoch"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SDAVT v3 training runs")
    parser.add_argument(
        "--log-dir",
        default="logs_sdavt_v3",
        help="Root log directory",
    )
    parser.add_argument(
        "--output",
        default="outputs_sdavt_v3/sdavt_v3_results_summary.csv",
    )
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    rows = []
    for metrics_csv in sorted(log_root.glob("*/metrics.csv")):
        rec = summarize_run(metrics_csv)
        if rec:
            rows.append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "val_epochs",
        "last_val_epoch",
        "last_val_accuracy",
        "last_val_f1",
        "best_val_accuracy",
        "best_val_accuracy_epoch",
        "best_val_f1",
        "best_val_f1_epoch",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summarized {len(rows)} runs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

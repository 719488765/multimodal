#!/usr/bin/env python3
"""按 phase 汇总消融实验结果，输出 fusion / m3 / modal 分表。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def to_float(val: str) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def summarize_run(metrics_csv: Path) -> Optional[Dict]:
    val_rows: List[Dict] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "val":
                continue
            val_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "accuracy": to_float(row.get("accuracy")),
                    "f1": to_float(row.get("f1")),
                }
            )
    if not val_rows:
        return None
    best_acc = max(val_rows, key=lambda r: r["accuracy"] or -1)
    best_f1 = max(val_rows, key=lambda r: r["f1"] or -1)
    last = val_rows[-1]
    return {
        "run": metrics_csv.parent.name,
        "epochs": len(val_rows),
        "best_acc": best_acc["accuracy"],
        "best_acc_ep": best_acc["epoch"],
        "best_f1": best_f1["f1"],
        "best_f1_ep": best_f1["epoch"],
        "last_acc": last["accuracy"],
        "last_f1": last["f1"],
    }


def infer_phase(run_name: str) -> str:
    u = run_name.upper()
    if "_F_M_" in u or "_F_C_" in u or "_F_O_" in u or "FUSION" in u:
        return "fusion"
    if "_M3_" in u:
        return "m3"
    if "_A_M_" in u:
        return "modal"
    if "_S3_" in u or "_S2_" in u or "_S1_" in u:
        return "baseline"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs_sdavt_v3")
    parser.add_argument("--output-dir", default="outputs_sdavt_v3")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_phase: Dict[str, List[Dict]] = defaultdict(list)
    for metrics in sorted(log_dir.glob("*/metrics.csv")):
        row = summarize_run(metrics)
        if not row:
            continue
        phase = infer_phase(row["run"])
        row["phase"] = phase
        by_phase[phase].append(row)

    for phase, rows in by_phase.items():
        out_path = out_dir / f"ablation_{phase}_table.csv"
        fields = [
            "run", "phase", "epochs",
            "best_acc", "best_acc_ep", "best_f1", "best_f1_ep",
            "last_acc", "last_f1",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda x: x["run"]):
                w.writerow({k: r.get(k) for k in fields})
        print(f"[OK] {out_path} ({len(rows)} runs)")

    # 融合榜：每数据集 best F1
    fusion_rows = by_phase.get("fusion", [])
    if fusion_rows:
        best_per_ds: Dict[str, Dict] = {}
        for r in fusion_rows:
            name = r["run"].upper()
            ds = "meld" if "MELD" in name else "crema" if "CREMA" in name else "mosei" if "MOSEI" in name else "?"
            if ds == "?":
                continue
            if ds not in best_per_ds or (r["best_f1"] or 0) > (best_per_ds[ds]["best_f1"] or 0):
                best_per_ds[ds] = r
        pick_path = out_dir / "ablation_fusion_winners.txt"
        with pick_path.open("w", encoding="utf-8") as f:
            for ds, r in sorted(best_per_ds.items()):
                f.write(f"{ds}: {r['run']} best_f1={r['best_f1']} @ep{r['best_f1_ep']}\n")
        print(f"[OK] {pick_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

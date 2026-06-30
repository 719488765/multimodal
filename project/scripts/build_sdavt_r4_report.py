#!/usr/bin/env python3
"""Refresh SDAVT R4 experiment report from queue + metrics.csv."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
STATUS_DIR = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "status"
DOCS_REPORT = PROJECT_ROOT / "docs" / "SDAVT_V3_R4_EXPERIMENT_RESULTS.md"


def _to_float(val: Any) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def read_best_from_csv(metrics_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[int], int]:
    val_rows: List[Dict[str, Any]] = []
    if not metrics_csv.exists():
        return None, None, None, 0
    with metrics_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "val":
                continue
            f1 = _to_float(row.get("f1"))
            if f1 is None:
                continue
            val_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "f1": f1,
                    "accuracy": _to_float(row.get("accuracy")),
                }
            )
    if not val_rows:
        return None, None, None, 0
    best_f1_row = max(val_rows, key=lambda r: r["f1"])
    best_acc = max((r["accuracy"] for r in val_rows if r["accuracy"] is not None), default=None)
    return best_f1_row["f1"], best_acc, best_f1_row["epoch"], len(val_rows)


def resolve_run_dir(job: Dict[str, Any]) -> Optional[str]:
    run_dir = job.get("run_dir")
    if run_dir and (LOG_DIR / run_dir / "metrics.csv").exists():
        return run_dir
    log_run = None
    cfg_path = PROJECT_ROOT / str(job.get("config", ""))
    if cfg_path.exists():
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        log_run = (cfg.get("experiment") or {}).get("log_run_dir")
    if log_run and (LOG_DIR / log_run / "metrics.csv").exists():
        return log_run
    prefix = f"SDAVT_R4_{job['id']}_"
    matches = sorted(p.name for p in LOG_DIR.glob(f"{prefix}*/metrics.csv"))
    if matches:
        return matches[-1].split("/")[0] if "/" in matches[-1] else Path(matches[-1]).parent.name
    matches = sorted(p.parent.name for p in LOG_DIR.glob(f"{prefix}*/metrics.csv"))
    return matches[-1] if matches else None


def enrich_jobs(jobs: List[Dict[str, Any]]) -> None:
    for job in jobs:
        run_dir = resolve_run_dir(job)
        if not run_dir:
            continue
        best_f1, best_acc, best_ep, epochs = read_best_from_csv(LOG_DIR / run_dir / "metrics.csv")
        job["run_dir"] = run_dir
        if best_f1 is not None:
            job["best_val_f1"] = round(best_f1, 4)
            job["best_val_acc"] = round(best_acc, 4) if best_acc is not None else None
            job["best_val_f1_ep"] = best_ep
            job["epochs_done"] = epochs


def tier2_flag(job: Dict[str, Any]) -> str:
    f1 = job.get("best_val_f1")
    acc = job.get("best_val_acc")
    tf1 = job.get("target_f1")
    tacc = job.get("target_acc")
    if f1 is None and acc is None:
        return "△"
    ok_f1 = tf1 is None or (f1 is not None and f1 >= float(tf1))
    ok_acc = tacc is None or (acc is not None and acc >= float(tacc))
    return "✓" if ok_f1 and ok_acc else "△"


def render_report(queue: Dict[str, Any]) -> str:
    jobs = queue.get("jobs", [])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c = Counter(j.get("status", "?") for j in jobs)
    lines = [
        f"# SDAVT v3 R4 实验结果汇总",
        "",
        f"*生成时间: {ts}*",
        "",
        "## 队列概览",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for k, v in sorted(c.items()):
        lines.append(f"| {k} | {v} |")

    phases = sorted({j.get("phase", "?") for j in jobs})
    for phase in phases:
        phase_jobs = [j for j in jobs if j.get("phase") == phase]
        if not phase_jobs:
            continue
        lines.extend(["", f"## {phase}", "", "| Job | Dataset | Status | Best F1@ep | Best Acc@ep | Tier-2 | Run |", "|-----|---------|--------|------------|-------------|--------|-----|"])
        for j in sorted(phase_jobs, key=lambda x: x.get("id", "")):
            f1 = j.get("best_val_f1")
            acc = j.get("best_val_acc")
            f1_ep = j.get("best_val_f1_ep", "—")
            f1_s = f"{f1:.4f} @ {f1_ep}" if f1 is not None else "— @ —"
            acc_s = f"{acc:.4f}" if acc is not None else "—"
            lines.append(
                f"| {j['id']} | {j.get('dataset','—')} | {j.get('status','?')} | "
                f"{f1_s} | {acc_s} | {tier2_flag(j)} | `{j.get('run_dir') or '—'}` |"
            )

    lines.extend(["", "---", "", "*本文档由 `scripts/build_sdavt_r4_report.py` 自动生成。*", ""])
    return "\n".join(lines)


def main() -> int:
    queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    enrich_jobs(queue["jobs"])
    QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = render_report(queue)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = STATUS_DIR / f"experiment_report_{ts}.md"
    out.write_text(report, encoding="utf-8")
    DOCS_REPORT.write_text(report, encoding="utf-8")
    latest = STATUS_DIR / "experiment_report_latest.md"
    shutil.copy2(out, latest)
    print(f"[OK] report -> {out}")
    print(f"[OK] docs   -> {DOCS_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

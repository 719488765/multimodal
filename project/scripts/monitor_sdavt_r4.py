#!/usr/bin/env python3
"""R4 实验进程监控：队列状态 + metrics 快照 + 进程文档数据刷新。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
STATUS_DIR = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "status"
TRACKER_MD = PROJECT_ROOT / "docs" / "SDAVT_V3_R4_EXPERIMENT_TRACKER.md"


def to_float(val: Any) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def summarize_metrics_csv(metrics_csv: Path) -> Optional[Dict[str, Any]]:
    val_rows: List[Dict[str, Any]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") != "val":
                continue
            val_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": to_float(row.get("loss")),
                    "accuracy": to_float(row.get("accuracy")),
                    "f1": to_float(row.get("f1")),
                    "cls_ce_unweighted": to_float(row.get("cls_ce_unweighted")),
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
        "last_cls_ce_uw": last["cls_ce_unweighted"],
    }


def scan_log_runs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not LOG_DIR.exists():
        return out
    for metrics in sorted(LOG_DIR.glob("*/metrics.csv")):
        rec = summarize_metrics_csv(metrics)
        if rec:
            out[rec["run"]] = rec
    return out


def load_queue() -> Dict[str, Any]:
    if not QUEUE_FILE.exists():
        return {"jobs": []}
    return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))


def enrich_queue_with_metrics(queue: Dict[str, Any], runs: Dict[str, Dict[str, Any]]) -> None:
    for job in queue.get("jobs", []):
        run_dir = job.get("run_dir") or ""
        if not run_dir:
            prefix = f"SDAVT_R4_{job['id']}_"
            matches = [n for n in runs if n.startswith(prefix)]
            if matches:
                run_dir = matches[0]
        if run_dir and run_dir in runs:
            r = runs[run_dir]
            job["best_val_f1"] = r["best_f1"]
            job["best_val_acc"] = r["best_acc"]
            job["best_val_f1_ep"] = r["best_f1_ep"]
            job["run_dir"] = run_dir


def print_status(queue: Dict[str, Any], runs: Dict[str, Dict[str, Any]]) -> None:
    jobs = queue.get("jobs", [])
    c = Counter(j.get("status", "?") for j in jobs)
    print(f"Queue: {QUEUE_FILE}")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    print(f"Log runs with metrics: {len(runs)}")
    print("--- running ---")
    for j in jobs:
        if j.get("status") == "running":
            print(f"  {j['id']} [{j.get('phase')}] gpu_hint={j.get('gpu_hint')}")
    print("--- pending (next 5) ---")
    pending = [j for j in jobs if j.get("status") == "pending"][:5]
    for j in pending:
        print(f"  {j['id']} [{j.get('phase')}]")


def write_status_json(queue: Dict[str, Any], runs: Dict[str, Any]) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = STATUS_DIR / f"snapshot_{ts}.json"
    payload = {
        "timestamp": ts,
        "queue_file": str(QUEUE_FILE),
        "log_dir": str(LOG_DIR),
        "run_count": len(runs),
        "queue_summary": Counter(j.get("status", "?") for j in queue.get("jobs", [])),
        "jobs": queue.get("jobs", []),
        "runs": runs,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = STATUS_DIR / "latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def update_tracker_appendix(queue: Dict[str, Any], runs: Dict[str, Any]) -> None:
    """在 tracker 文档末尾刷新「自动快照」区块（不覆盖手工计划部分）。"""
    if not TRACKER_MD.exists():
        return
    text = TRACKER_MD.read_text(encoding="utf-8")
    marker = "<!-- AUTO_SNAPSHOT_START -->"
    end_marker = "<!-- AUTO_SNAPSHOT_END -->"
    if marker not in text:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        marker,
        "",
        f"## 自动快照（{ts}）",
        "",
        "### 队列状态",
        "",
    ]
    c = Counter(j.get("status", "?") for j in queue.get("jobs", []))
    for k, v in sorted(c.items()):
        lines.append(f"- **{k}**: {v}")
    lines.extend(["", "### 已完成任务指标", "", "| Job | Phase | Best F1@ep | Best Acc@ep | Run |", "|-----|-------|------------|-------------|-----|"])
    for j in sorted(queue.get("jobs", []), key=lambda x: (x.get("phase", ""), x.get("id", ""))):
        if j.get("status") != "done":
            continue
        lines.append(
            f"| {j['id']} | {j.get('phase','')} | "
            f"{j.get('best_val_f1','—')} @ {j.get('best_val_f1_ep','—')} | "
            f"{j.get('best_val_acc','—')} @ {j.get('best_val_acc_ep','—')} | "
            f"`{j.get('run_dir','')}` |"
        )
    lines.extend(["", "### 日志目录 run 数", "", f"- `logs_sdavt_v3_r4/`: **{len(runs)}** runs with metrics.csv", ""])
    lines.append(end_marker)

    head = text.split(marker)[0].rstrip()
    tail_parts = text.split(end_marker)
    tail = tail_parts[1] if len(tail_parts) > 1 else ""
    TRACKER_MD.write_text("\n".join(lines) + tail, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Print status and write snapshot")
    args = parser.parse_args()

    queue = load_queue()
    runs = scan_log_runs()
    enrich_queue_with_metrics(queue, runs)

    if args.once or True:
        print_status(queue, runs)
        snap = write_status_json(queue, runs)
        print(f"[OK] snapshot -> {snap}")
        update_tracker_appendix(queue, runs)
        if TRACKER_MD.exists():
            print(f"[OK] tracker updated -> {TRACKER_MD}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

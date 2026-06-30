#!/usr/bin/env python3
"""将 R4 队列中指定 job 重置为 pending（用于 failed 重跑）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_ids", nargs="+", help="Job IDs to reset, e.g. M3_M1_roberta")
    parser.add_argument("--phase", default="", help="Optional phase filter")
    args = parser.parse_args()

    q = json.loads(QUEUE.read_text(encoding="utf-8"))
    now = datetime.now().isoformat(timespec="seconds")
    reset = 0
    for j in q.get("jobs", []):
        if j["id"] not in args.job_ids:
            continue
        if args.phase and j.get("phase") != args.phase:
            continue
        j["status"] = "pending"
        j["started_at"] = None
        j["finished_at"] = None
        j["run_dir"] = None
        j["best_val_f1"] = None
        j["best_val_acc"] = None
        j["updated_at"] = now
        j["note"] = "rerun_after_fix"
        reset += 1
        print(f"[OK] reset {j.get('phase')}/{j['id']} -> pending")
    if not reset:
        print("[WARN] no jobs matched")
        return 1
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] queue updated ({reset} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

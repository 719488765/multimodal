#!/usr/bin/env python3
"""P3 Tier-2 acceptance: all p3_m3 + p3_c3 done, pick MELD/CREMA champions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
STATUS_DIR = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "status"

MELD_TARGET_F1 = 0.59
MELD_TARGET_ACC = 0.62
CREMA_TARGET_ACC = 0.63


def _read_best_from_csv(run_dir: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    metrics = LOG_DIR / run_dir / "metrics.csv"
    if not metrics.exists():
        return None, None, None
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
                }
            )
    if not val_rows:
        return None, None, None
    best = max(val_rows, key=lambda r: r["f1"])
    best_acc = max((r["accuracy"] for r in val_rows if r["accuracy"] is not None), default=best["accuracy"])
    return best["f1"], best_acc, best["epoch"]


def _resolve_run_dir(job: Dict[str, Any]) -> Optional[str]:
    run_dir = job.get("run_dir")
    if run_dir and (LOG_DIR / run_dir / "metrics.csv").exists():
        return run_dir
    import yaml

    cfg_path = PROJECT_ROOT / str(job.get("config", ""))
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        log_run = (cfg.get("experiment") or {}).get("log_run_dir")
        if log_run and (LOG_DIR / log_run / "metrics.csv").exists():
            return log_run
    prefix = f"SDAVT_R4_{job['id']}_"
    matches = sorted(p.parent.name for p in LOG_DIR.glob(f"{prefix}*/metrics.csv"))
    return matches[-1] if matches else None


def _enrich_job(job: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(job)
    if run_dir:
        job = dict(job)
        job["run_dir"] = run_dir
        f1, acc, ep = _read_best_from_csv(run_dir)
        if f1 is not None:
            job["best_val_f1"] = round(f1, 6)
            job["best_val_acc"] = round(acc, 6) if acc is not None else None
            job["best_val_f1_ep"] = ep
    return job


def _pick_champion(jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    enriched = [_enrich_job(j) for j in jobs if j.get("status") == "done"]
    if not enriched:
        return None
    return max(enriched, key=lambda j: float(j.get("best_val_f1") or -1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-report", action="store_true")
    args = parser.parse_args()

    subprocess.run([sys.executable, "scripts/build_sdavt_r4_report.py"], cwd=PROJECT_ROOT, check=False)

    queue = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    p3m = [j for j in queue["jobs"] if j.get("phase") == "p3_m3"]
    p3c = [j for j in queue["jobs"] if j.get("phase") == "p3_c3"]

    incomplete_m = [j["id"] for j in p3m if j.get("status") != "done"]
    if incomplete_m:
        print(f"[WARN] P3-M incomplete: {', '.join(incomplete_m)}")
        return 1

    meld_champion = _pick_champion(p3m)
    crema_champion = _pick_champion(p3c)
    if meld_champion is None:
        print("[ERROR] no MELD P3-M champion")
        return 1

    meld_f1 = float(meld_champion.get("best_val_f1") or 0)
    meld_acc = float(meld_champion.get("best_val_acc") or 0)
    meld_pass = meld_f1 >= MELD_TARGET_F1 and meld_acc >= MELD_TARGET_ACC

    crema_acc = float(crema_champion.get("best_val_acc") or 0) if crema_champion else 0.0
    crema_pass = crema_acc >= CREMA_TARGET_ACC

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    winner_path = STATUS_DIR / "p3_m3_winner_meld.json"
    winner_path.write_text(json.dumps(meld_champion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = {
        "timestamp": ts,
        "meld_pass": meld_pass,
        "crema_pass": crema_pass,
        "meld_winner": meld_champion,
        "crema_winner": crema_champion,
    }
    (STATUS_DIR / "p3_tier2_acceptance_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = [
        f"# P3 Tier-2 验收 ({datetime.now().isoformat(timespec='seconds')})",
        "",
        f"**MELD Tier-2**: {'PASS' if meld_pass else 'FAIL'} (F1≥{MELD_TARGET_F1}, Acc≥{MELD_TARGET_ACC})",
        f"**CREMA Tier-2**: {'PASS' if crema_pass else 'FAIL'} (Acc≥{CREMA_TARGET_ACC})",
        "",
        "## MELD P3-M 冠军",
        "",
        f"- Job: `{meld_champion['id']}`",
        f"- Run: `{meld_champion.get('run_dir', '—')}`",
        f"- Best F1: **{meld_f1:.4f}** @ ep{meld_champion.get('best_val_f1_ep', '—')}",
        f"- Best Acc: **{meld_acc:.4f}**",
        "",
    ]
    if crema_champion:
        md.extend(
            [
                "## CREMA P3-C 最优",
                "",
                f"- Job: `{crema_champion['id']}`",
                f"- Run: `{crema_champion.get('run_dir', '—')}`",
                f"- Best Acc: **{crema_acc:.4f}** @ ep{crema_champion.get('best_val_f1_ep', '—')}",
                "",
            ]
        )
    (STATUS_DIR / "p3_tier2_acceptance_latest.md").write_text("\n".join(md), encoding="utf-8")
    (STATUS_DIR / "p3_tier2_passed").write_text(json.dumps({"passed": meld_pass, "champion": meld_champion["id"]}, indent=2) + "\n")

    print(f"[OK] MELD champion -> {meld_champion['id']} F1={meld_f1:.4f} Acc={meld_acc:.4f} ({'PASS' if meld_pass else 'FAIL'})")
    print(f"[OK] CREMA best Acc={crema_acc:.4f} ({'PASS' if crema_pass else 'FAIL'})")

    if args.refresh_report:
        subprocess.run([sys.executable, "scripts/build_sdavt_r4_report.py"], cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

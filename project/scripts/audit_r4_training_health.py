#!/usr/bin/env python3
"""Audit SDAVT R4 training health: collapse signatures, MD5 duplicates, Tier-2 gaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs_sdavt_v3_r4"
CKPT_DIR = PROJECT_ROOT / "checkpoints_sdavt_v3_r4"
QUEUE_FILE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
OUT_JSON = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "tables" / "r4_training_audit.json"

TIER2_TARGETS = {
    "meld": {"f1": 0.59, "acc": 0.62},
    "mosei": {"f1": 0.67, "acc": None},
    "crema": {"f1": None, "acc": 0.63},
}

DATASET_CLASSES = {"crema": 6, "mosei": 7, "meld": 7}
COLLAPSE_EPS = 0.025
COLLAPSE_F1_MAX = 0.15


def _md5(path: Path, max_bytes: int = 0) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.md5()
    if max_bytes <= 0:
        h.update(path.read_bytes())
        return h.hexdigest()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    st = path.stat()
    h.update(str(st.st_size).encode())
    return h.hexdigest()


def _read_metrics(path: Path) -> Dict[str, Any]:
    val_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return {"val_rows": [], "train_rows": []}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase = row.get("phase")
            rec = dict(row)
            if phase == "val" and row.get("f1"):
                rec["f1"] = float(row["f1"])
                rec["accuracy"] = float(row["accuracy"]) if row.get("accuracy") else None
                rec["epoch"] = int(row["epoch"])
                val_rows.append(rec)
            elif phase == "train" and row.get("loss"):
                rec["loss"] = float(row["loss"])
                rec["epoch"] = int(row["epoch"])
                train_rows.append(rec)
    return {"val_rows": val_rows, "train_rows": train_rows}


def _uniform_f1(num_classes: int) -> float:
    return math.log(num_classes) / num_classes


def _detect_dataset(run_dir: str) -> Optional[str]:
    name = run_dir.upper()
    if "_C_" in name or name.endswith("_F_C_ES") or "CREMA" in name or name.startswith("SDAVT_R4_C3"):
        return "crema"
    if "_O_" in name or name.startswith("SDAVT_R4_F_O"):
        return "mosei"
    if "_M_" in name or name.startswith("SDAVT_R4_F_M") or name.startswith("SDAVT_R4_M3"):
        return "meld"
    return None


def _resolve_queue_run_dir(job: Dict[str, Any]) -> Optional[str]:
    run_dir = job.get("run_dir")
    if run_dir and (LOG_DIR / run_dir / "metrics.csv").is_file():
        return run_dir
    cfg_path = PROJECT_ROOT / str(job.get("config", ""))
    if cfg_path.is_file():
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        log_run = (cfg.get("experiment") or {}).get("log_run_dir")
        if log_run and (LOG_DIR / log_run / "metrics.csv").is_file():
            return log_run
    prefix = f"SDAVT_R4_{job['id']}_"
    matches = sorted(p.parent.name for p in LOG_DIR.glob(f"{prefix}*/metrics.csv"))
    if matches:
        return matches[-1]
    fallback = f"SDAVT_R4_{job['id']}"
    return fallback if (LOG_DIR / fallback / "metrics.csv").is_file() else None


# Known baseline slot reused for P2 comparison (not a training collision).
_DUPE_IGNORE_JOB_SETS = [
    frozenset({"R4_B_O0", "F_O_ES"}),
]
_PHASE_PRIORITY = {
    "p4_modal": 5,
    "p3_m3": 4,
    "p3_c_plus": 4,
    "p2_fusion": 3,
    "p1_baseline": 2,
    "p0_fix": 1,
}


def _queue_only_p0(collapses: List[Dict[str, Any]], duplicates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not QUEUE_FILE.is_file():
        return collapses, duplicates

    import importlib.util

    validate_path = PROJECT_ROOT / "scripts" / "validate_p4_job_metrics.py"
    spec = importlib.util.spec_from_file_location("validate_p4_job_metrics", validate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {validate_path}")
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    validate_job = v4.validate_job

    q = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    queue_collapses: List[Dict[str, Any]] = []
    job_run: Dict[str, str] = {}
    job_meta: Dict[str, Dict[str, Any]] = {}

    ranked_jobs = sorted(
        q.get("jobs", []),
        key=lambda j: (_PHASE_PRIORITY.get(str(j.get("phase", "")), 0), str(j.get("updated_at", ""))),
    )
    for job in ranked_jobs:
        run_dir = _resolve_queue_run_dir(job)
        if not run_dir:
            continue
        job_run[job["id"]] = run_dir
        job_meta[job["id"]] = job

    for job_id, run_dir in job_run.items():
        job = job_meta[job_id]
        if job.get("phase") != "p4_modal":
            continue
        dataset = str(job.get("dataset", "")).lower()
        if validate_job(job_id, dataset, run_dir) != 0:
            match = next((c for c in collapses if c["run_dir"] == run_dir), None)
            queue_collapses.append(
                match
                or {
                    "run_dir": run_dir,
                    "job_id": job["id"],
                    "dataset": dataset,
                    "reasons": ["validate_p4_job_metrics failed"],
                    "severity": "P0",
                }
            )

    md5_jobs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for job_id, run_dir in job_run.items():
        digest = _md5(LOG_DIR / run_dir / "metrics.csv")
        if digest:
            md5_jobs[digest].append((job_id, run_dir))

    queue_dupes: List[Dict[str, Any]] = []
    for digest, items in md5_jobs.items():
        if len(items) < 2:
            continue
        job_ids = frozenset(j for j, _ in items)
        if any(job_ids == ignore for ignore in _DUPE_IGNORE_JOB_SETS):
            continue
        queue_dupes.append(
            {
                "kind": "metrics.csv",
                "md5": digest,
                "jobs": [list(x) for x in items],
                "runs": [rd for _, rd in items],
                "severity": "P0",
            }
        )

    return queue_collapses, queue_dupes


def _collapse_info(run_dir: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    val_rows = metrics["val_rows"]
    train_rows = metrics["train_rows"]
    if not val_rows:
        return None
    best_f1 = max(r["f1"] for r in val_rows)
    ep0_f1 = next((r["f1"] for r in val_rows if r["epoch"] == 0), None)
    dataset = _detect_dataset(run_dir)
    num_classes = DATASET_CLASSES.get(dataset or "", 7)
    uniform = _uniform_f1(num_classes)

    reasons: List[str] = []
    if best_f1 <= COLLAPSE_F1_MAX:
        reasons.append(f"best_f1={best_f1:.4f}<={COLLAPSE_F1_MAX}")
    if abs(best_f1 - uniform) <= COLLAPSE_EPS:
        reasons.append(f"best_f1≈ln({num_classes})/{num_classes}={uniform:.4f}")
    if ep0_f1 is not None and dataset == "mosei" and ep0_f1 < 0.15:
        reasons.append(f"mosei_ep0_f1={ep0_f1:.4f}<0.15")

    early_train = [r for r in train_rows if r["epoch"] < 10]
    if early_train and all(r["loss"] == 0.0 for r in early_train[:8]):
        reasons.append("train_loss_zero_early_epochs")

    if not reasons:
        return None
    return {
        "run_dir": run_dir,
        "dataset": dataset,
        "best_f1": round(best_f1, 6),
        "ep0_f1": round(ep0_f1, 6) if ep0_f1 is not None else None,
        "uniform_f1": round(uniform, 6),
        "reasons": reasons,
        "severity": "P0",
    }


def _find_duplicates() -> List[Dict[str, Any]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for metrics in sorted(LOG_DIR.glob("*/metrics.csv")):
        digest = _md5(metrics)
        if digest:
            groups[digest].append(metrics.parent.name)

    ckpt_groups: Dict[str, List[str]] = defaultdict(list)
    for ckpt in sorted(CKPT_DIR.glob("*/checkpoint_pretrain_best_f1.pth")):
        digest = _md5(ckpt, max_bytes=65536)
        if digest:
            ckpt_groups[digest].append(ckpt.parent.name)

    dupes: List[Dict[str, Any]] = []
    for digest, runs in groups.items():
        if len(runs) > 1:
            dupes.append({"kind": "metrics.csv", "md5": digest, "runs": sorted(runs), "severity": "P0"})
    for digest, runs in ckpt_groups.items():
        if len(runs) > 1:
            dupes.append({"kind": "checkpoint_pretrain_best_f1.pth", "md5": digest, "runs": sorted(runs), "severity": "P0"})
    return dupes


def _tier2_champions() -> Dict[str, Dict[str, Any]]:
    """Best done job per dataset from queue + logs."""
    queue_path = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
    champions: Dict[str, Dict[str, Any]] = {d: {"job_id": None, "best_f1": None, "best_acc": None} for d in TIER2_TARGETS}

    if queue_path.is_file():
        q = json.loads(queue_path.read_text(encoding="utf-8"))
        for job in q.get("jobs", []):
            if job.get("status") != "done":
                continue
            ds = job.get("dataset")
            if ds not in champions:
                continue
            f1 = job.get("best_val_f1")
            acc = job.get("best_val_acc")
            cur = champions[ds]
            score = f1 if ds in ("meld", "mosei") else acc
            cur_score = cur["best_f1"] if ds in ("meld", "mosei") else cur["best_acc"]
            if score is not None and (cur_score is None or score > cur_score):
                champions[ds] = {
                    "job_id": job["id"],
                    "best_f1": f1,
                    "best_acc": acc,
                    "run_dir": job.get("run_dir"),
                }

    # Also scan P2 fusion winners file
    winners_path = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "tables" / "r4_fusion_winners.json"
    if winners_path.is_file():
        winners = json.loads(winners_path.read_text(encoding="utf-8"))
        for ds_key, entry in winners.items():
            ds = ds_key.replace("_crema", "crema").replace("_mosei", "mosei").replace("_meld", "meld")
            if ds not in champions:
                continue
            f1 = entry.get("best_f1") or entry.get("f1")
            acc = entry.get("best_acc") or entry.get("acc")
            cur = champions[ds]
            if ds in ("meld", "mosei") and f1 is not None:
                if cur["best_f1"] is None or f1 > cur["best_f1"]:
                    champions[ds] = {"job_id": entry.get("job_id"), "best_f1": f1, "best_acc": acc, "run_dir": entry.get("run_dir")}
            elif ds == "crema" and acc is not None:
                if cur["best_acc"] is None or acc > cur["best_acc"]:
                    champions[ds] = {"job_id": entry.get("job_id"), "best_f1": f1, "best_acc": acc, "run_dir": entry.get("run_dir")}

    tier2: Dict[str, Dict[str, Any]] = {}
    for ds, targets in TIER2_TARGETS.items():
        ch = champions[ds]
        ok_f1 = targets["f1"] is None or (ch["best_f1"] is not None and ch["best_f1"] >= targets["f1"])
        ok_acc = targets["acc"] is None or (ch["best_acc"] is not None and ch["best_acc"] >= targets["acc"])
        tier2[ds] = {
            "champion": ch,
            "targets": targets,
            "pass": ok_f1 and ok_acc,
            "severity": None if (ok_f1 and ok_acc) else "P2",
        }
    return tier2


def run_audit(queue_only: bool = False) -> Dict[str, Any]:
    collapses: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []

    for metrics_path in sorted(LOG_DIR.glob("*/metrics.csv")):
        run_dir = metrics_path.parent.name
        metrics = _read_metrics(metrics_path)
        if not metrics["val_rows"]:
            continue
        best_f1 = max(r["f1"] for r in metrics["val_rows"])
        best_acc = max((r["accuracy"] for r in metrics["val_rows"] if r["accuracy"] is not None), default=None)
        run_summaries.append({"run_dir": run_dir, "best_f1": round(best_f1, 6), "best_acc": round(best_acc, 6) if best_acc else None})
        info = _collapse_info(run_dir, metrics)
        if info:
            collapses.append(info)

    duplicates = _find_duplicates()
    tier2 = _tier2_champions()

    canonical: List[str] = []
    if queue_only:
        collapses, duplicates = _queue_only_p0(collapses, duplicates)
        if QUEUE_FILE.is_file():
            q = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
            canonical = sorted(
                rd for job in q.get("jobs", []) if (rd := _resolve_queue_run_dir(job))
            )

    p0_count = len(collapses) + len(duplicates)
    p2_count = sum(1 for v in tier2.values() if not v["pass"])

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "queue_only" if queue_only else "all_logs",
        "summary": {
            "total_runs_with_metrics": len(run_summaries),
            "collapse_count": len(collapses),
            "duplicate_group_count": len(duplicates),
            "tier2_fail_count": p2_count,
            "p0_issues": p0_count,
        },
        "collapses": collapses,
        "duplicates": duplicates,
        "tier2": tier2,
        "runs": run_summaries,
    }
    if queue_only:
        report["canonical_run_dirs"] = sorted(canonical)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_JSON))
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any P0 issues")
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="Count P0 only on canonical experiment_queue.json log slots",
    )
    args = parser.parse_args()

    report = run_audit(queue_only=args.queue_only)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    s = report["summary"]
    scope = report.get("scope", "all_logs")
    print(f"[audit] scope={scope} runs={s['total_runs_with_metrics']} collapses={s['collapse_count']} "
          f"dupes={s['duplicate_group_count']} tier2_fail={s['tier2_fail_count']} p0={s['p0_issues']}")
    print(f"[OK] -> {out_path}")

    if args.strict and s["p0_issues"] > 0:
        print("[FAIL] P0 issues remain")
        for c in report.get("collapses", []):
            print(f"  collapse: {c.get('job_id') or c.get('run_dir')} {c.get('reasons')}")
        for d in report.get("duplicates", []):
            print(f"  duplicate: {d.get('jobs') or d.get('runs')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

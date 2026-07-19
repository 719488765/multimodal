#!/usr/bin/env python3
"""Prune R4 TensorBoard logs: archive failed retries / empty runs outside logdir."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs_sdavt_v3_r4"
ARCHIVE_DIR = ROOT / "logs_sdavt_v3_r4_archived"
QUEUE = ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
STATUS = ROOT / "outputs_sdavt_v3_r4" / "status"

# Always keep queue runs + successful close-out (exclude failed C4_C1)
KEEP_EXTRA = {
    "SDAVT_R4_C4_C2_c3_base_acc",
    "SDAVT_R4_C4_C3_c3_warmstart_acc",
    "SDAVT_R4_M3_M7_chinese_agent",
}

# Always archive (failed / known bad)
FORCE_ARCHIVE = {
    "SDAVT_R4_C4_C1_combo_acc",
}

# MOSEI early aborted retries (F1≈0.088 @ 1–3 epochs); canonical runs kept in queue
MOSEI_RETRY_PREFIXES = (
    "SDAVT_R4_F_O_LFT_20260622_174508",
    "SDAVT_R4_F_O_LFT_20260622_212841",
    "SDAVT_R4_F_O_STD_20260622_191324",
    "SDAVT_R4_F_O_STD_20260622_213550",
    "SDAVT_R4_F_O_TS_20260622_202016",
    "SDAVT_R4_F_O_TS_20260622_213845",
)


def val_epoch_count(run_dir: Path) -> int:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        return 0
    return sum(
        1
        for row in csv.DictReader(csv_path.open(encoding="utf-8"))
        if row.get("phase") == "val" and row.get("f1")
    )


def load_keep_set() -> set[str]:
    data = json.loads(QUEUE.read_text(encoding="utf-8"))
    keep = {j["run_dir"] for j in data.get("jobs", []) if j.get("run_dir")}
    keep |= KEEP_EXTRA
    return keep


def plan_archive(dry_run: bool = True) -> dict:
    keep = load_keep_set()
    all_runs = sorted(p.name for p in LOG_DIR.iterdir() if p.is_dir() and not p.name.startswith("_"))
    to_archive: list[dict] = []
    to_keep: list[str] = []

    for name in all_runs:
        if name in keep:
            to_keep.append(name)
            continue
        reason = "not_in_queue"
        if name in FORCE_ARCHIVE:
            reason = "failed_nan"
        elif name in MOSEI_RETRY_PREFIXES:
            reason = "mosei_p0_aborted_retry"
        elif val_epoch_count(LOG_DIR / name) == 0:
            reason = "empty_metrics"
        elif val_epoch_count(LOG_DIR / name) <= 1:
            reason = "single_val_epoch_abort"
        to_archive.append({"run_dir": name, "reason": reason, "val_epochs": val_epoch_count(LOG_DIR / name)})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_dir": str(LOG_DIR),
        "archive_dir": str(ARCHIVE_DIR),
        "keep_count": len(to_keep),
        "archive_count": len(to_archive),
        "keep": sorted(to_keep),
        "archive": to_archive,
        "dry_run": dry_run,
    }
    return manifest


def apply_archive(manifest: dict) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for item in manifest["archive"]:
        src = LOG_DIR / item["run_dir"]
        dst = ARCHIVE_DIR / item["run_dir"]
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"[archived] {item['run_dir']} ({item['reason']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune R4 TB logs into logs_sdavt_v3_r4_archived/")
    parser.add_argument("--apply", action="store_true", help="Move runs (default: dry-run only)")
    args = parser.parse_args()

    manifest = plan_archive(dry_run=not args.apply)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = STATUS / f"tb_cleanup_manifest_{ts}.json"
    STATUS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] manifest -> {out}")
    print(f"keep={manifest['keep_count']} archive={manifest['archive_count']} dry_run={manifest['dry_run']}")
    for item in manifest["archive"]:
        print(f"  -> {item['run_dir']} ({item['reason']}, val={item['val_epochs']})")

    if args.apply:
        apply_archive(manifest)
        print(f"[OK] archived to {ARCHIVE_DIR}")
    else:
        print("Dry-run only. Re-run with --apply to move runs.")


if __name__ == "__main__":
    main()

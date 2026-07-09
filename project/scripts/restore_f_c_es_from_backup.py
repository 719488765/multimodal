#!/usr/bin/env python3
"""Restore F_C_ES slot from archived pre-antiof backup (valid F1>=0.54, not AVT collision)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "archived" / "p2_crema_es_pre_antiof_20260625_175712"
LOG_SLOT = PROJECT_ROOT / "logs_sdavt_v3_r4" / "SDAVT_R4_F_C_ES"
CKPT_SLOT = PROJECT_ROOT / "checkpoints_sdavt_v3_r4" / "SDAVT_R4_F_C_ES"
AVT_METRICS = PROJECT_ROOT / "logs_sdavt_v3_r4" / "SDAVT_R4_R4_A_C_AVT" / "metrics.csv"
STATUS_DIR = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "status"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _find_backup() -> tuple[Path, Path]:
    log_src = ckpt_src = None
    for p in ARCHIVE.iterdir():
        if p.name.startswith("logs_sdavt_v3_r4_SDAVT_R4_F_C_ES"):
            log_src = p
        elif p.name.startswith("checkpoints_sdavt_v3_r4_SDAVT_R4_F_C_ES"):
            ckpt_src = p
    if not log_src or not ckpt_src:
        raise SystemExit(f"backup not found under {ARCHIVE}")
    return log_src, ckpt_src


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log_src, ckpt_src = _find_backup()
    metrics_src = log_src / "metrics.csv"
    if not metrics_src.is_file():
        raise SystemExit(f"missing metrics in {log_src}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "archived" / f"f_c_es_pre_restore_{ts}"
    arch.mkdir(parents=True, exist_ok=True)

    for slot in (LOG_SLOT, CKPT_SLOT):
        if slot.is_dir():
            shutil.move(str(slot), str(arch / slot.name))

    shutil.copytree(log_src, LOG_SLOT)
    shutil.copytree(ckpt_src, CKPT_SLOT)

    restored_metrics = LOG_SLOT / "metrics.csv"
    if AVT_METRICS.is_file() and _md5(restored_metrics) == _md5(AVT_METRICS):
        raise SystemExit("restored metrics still identical to AVT collision")

    meta = {
        "job_id": "F_C_ES",
        "experiment_name": "SDAVT_R4_F_C_ES",
        "log_run_dir": "SDAVT_R4_F_C_ES",
        "config_path": "config/sdavt_v3_r4/p2_fusion/crema/F_C_ES_emotion_shift.yaml",
        "restored_from": str(log_src),
        "restored_at": datetime.now().isoformat(timespec="seconds"),
        "replace_log_dir": True,
    }
    (LOG_SLOT / ".run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/finalize_r4_p2_retrain.py", "F_C_ES", "--status", "done", "--note", f"restored_from_backup_{ts}"],
        cwd=PROJECT_ROOT,
        check=True,
    )

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (STATUS_DIR / "f_c_es_isolated_retrain_done").write_text(f"restored {ts}\n", encoding="utf-8")
    print(f"[OK] F_C_ES restored from {log_src.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

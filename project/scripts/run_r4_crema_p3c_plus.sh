#!/usr/bin/env bash
# CREMA P3-C+ Tier-2 optimization: target Acc >= 0.63
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

CFG="config/sdavt_v3_r4/p3_c_plus/crema/C4_C1_combo_acc.yaml"
RUN="SDAVT_R4_C4_C1_combo_acc"
STATUS_DIR="outputs_sdavt_v3_r4/status"
TARGET_ACC=0.63
GPU="1"

mkdir -p "$STATUS_DIR"
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

echo ">>> CREMA P3-C+ C4_C1_combo_acc on GPU${GPU}"
python3 scripts/train.py --config "$CFG" --mode pretrain --replace_log_dir \
  >> "$STATUS_DIR/crema_p3c_plus.log" 2>&1

python3 - <<PY
import csv
from pathlib import Path
p = Path("logs_sdavt_v3_r4/$RUN/metrics.csv")
best_acc = None
best_f1 = None
with p.open() as f:
    for r in csv.DictReader(f):
        if r.get("phase") != "val":
            continue
        if r.get("accuracy"):
            v = float(r["accuracy"])
            best_acc = v if best_acc is None else max(best_acc, v)
        if r.get("f1"):
            v = float(r["f1"])
            best_f1 = v if best_f1 is None else max(best_f1, v)
print(f"best_acc={best_acc} best_f1={best_f1} target_acc=$TARGET_ACC")
if best_acc is None or best_acc < $TARGET_ACC:
    raise SystemExit(f"CREMA P3-C+ Acc {best_acc} < $TARGET_ACC")
print("[OK] CREMA Tier-2 Acc target met")
PY

touch "$STATUS_DIR/crema_p3c_plus_done"
echo "[OK] CREMA P3-C+ complete"

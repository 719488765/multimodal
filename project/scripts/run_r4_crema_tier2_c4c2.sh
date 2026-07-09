#!/usr/bin/env bash
# CREMA Tier-2 round 2: C4_C2 (C3_C2 baseline + Acc-oriented early-stop).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"
STATUS="outputs_sdavt_v3_r4/status"
CFG="config/sdavt_v3_r4/p3_c_plus/crema/C4_C2_c3_base_acc.yaml"
LOG="$STATUS/crema_tier2_c4c2_live.log"

mkdir -p "$STATUS"
echo "[$(date '+%F %T')] CREMA Tier-2 C4_C2 start GPU=${CUDA_VISIBLE_DEVICES:-1}" | tee -a "$LOG"

"$PY" scripts/train.py --config "$CFG" --mode pretrain --replace_log_dir 2>&1 | tee -a "$LOG"

# Tier-2 gate: Acc >= 0.63
best_acc="$("$PY" - <<'PY'
import csv
from pathlib import Path
p = Path("logs_sdavt_v3_r4/SDAVT_R4_C4_C2_c3_base_acc/metrics.csv")
best = None
if p.is_file():
    for r in csv.DictReader(p.open()):
        if r.get("phase") == "val" and r.get("accuracy"):
            v = float(r["accuracy"])
            best = v if best is None else max(best, v)
print(best if best is not None else "nan")
PY
)"

echo "[$(date '+%F %T')] best_val_acc=$best_acc" | tee -a "$LOG"
"$PY" - <<PY
import math
v=float("$best_acc")
if math.isnan(v) or v < 0.63:
    raise SystemExit(1)
PY

touch "$STATUS/crema_tier2_c4c2_passed"
echo "[OK] CREMA Tier-2 C4_C2 Acc>=0.63" | tee -a "$LOG"

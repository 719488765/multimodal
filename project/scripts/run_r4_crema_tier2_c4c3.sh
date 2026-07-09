#!/usr/bin/env bash
# CREMA Tier-2 round 3: warm-start from C3_C2 champion, Acc-oriented fine-tune.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"
STATUS="outputs_sdavt_v3_r4/status"
CFG="config/sdavt_v3_r4/p3_c_plus/crema/C4_C3_c3_warmstart_acc.yaml"
LOG="$STATUS/crema_tier2_c4c3_live.log"
RESUME="checkpoints_sdavt_v3_r4/SDAVT_R4_C3_C2_w2v_large_20260626_004150/checkpoint_pretrain_best_f1.pth"

if [[ ! -f "$RESUME" ]]; then
  echo "[FAIL] C3_C2 resume ckpt missing: $RESUME"
  exit 1
fi

mkdir -p "$STATUS"
echo "[$(date '+%F %T')] CREMA Tier-2 C4_C3 warm-start from C3_C2 GPU=${CUDA_VISIBLE_DEVICES:-1}" | tee -a "$LOG"
echo "resume=$RESUME" | tee -a "$LOG"

"$PY" scripts/train.py --config "$CFG" --mode pretrain --replace_log_dir \
  --resume "$RESUME" 2>&1 | tee -a "$LOG"

best_acc="$("$PY" - <<'PY'
import csv
from pathlib import Path
p = Path("logs_sdavt_v3_r4/SDAVT_R4_C4_C3_c3_warmstart_acc/metrics.csv")
best = None
if p.is_file():
    for r in csv.DictReader(p.open()):
        if r.get("phase") == "val" and r.get("accuracy"):
            v = float(r["accuracy"])
            best = v if best is None else max(best, v)
print(best if best is not None else "nan")
PY
)"

echo "[$(date '+%F %T')] best_val_acc=$best_acc (C3_C2 baseline=0.5672)" | tee -a "$LOG"

"$PY" - <<PY
import math
v = float("$best_acc")
if math.isnan(v):
    raise SystemExit(1)
if v >= 0.63:
    print("[PASS] Tier-2 Acc>=0.63")
elif v > 0.5672:
    print("[PARTIAL] beat C3_C2 but below 0.63")
else:
    print("[FAIL] did not beat C3_C2 champion")
    raise SystemExit(1)
PY

if python3 -c "exit(0 if float('$best_acc')>=0.63 else 1)"; then
  touch "$STATUS/crema_tier2_c4c3_passed"
fi
echo "[OK] C4_C3 complete best_acc=$best_acc" | tee -a "$LOG"

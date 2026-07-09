#!/usr/bin/env bash
# Rerun failed MELD P4 R4_A_M_V (video-only; leader_modal=video fix).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"

ARCH="outputs_sdavt_v3_r4/archived/p4_meld_v_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH"
JOB="R4_A_M_V"
RUN="SDAVT_R4_${JOB}"

for base in logs_sdavt_v3_r4 checkpoints_sdavt_v3_r4; do
  d="$base/$RUN"
  if [[ -d "$d" ]]; then
    rm -rf "$ARCH/$(basename "$d")"
    mv "$d" "$ARCH/" && echo "archived $d"
  fi
done

"$PY" scripts/reset_r4_queue_jobs.py "$JOB" --phase p4_modal

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cfg="$("$PY" - <<PY
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
for j in q["jobs"]:
    if j["id"]=="$JOB":
        print(j["config"]); break
PY
)"
echo ">>> $JOB $cfg"
"$PY" scripts/train.py --config "$cfg" --mode pretrain --replace_log_dir

if ! "$PY" scripts/validate_p4_job_metrics.py --dataset meld "$JOB"; then
  echo "[FAIL] validate_p4_job_metrics collapse check failed"
  exit 1
fi

"$PY" - <<'PY'
import csv
import sys
from pathlib import Path

metrics = Path("logs_sdavt_v3_r4/SDAVT_R4_R4_A_M_V/metrics.csv")
if not metrics.is_file():
    print(f"[FAIL] missing {metrics}")
    sys.exit(1)

ep0 = best = None
with metrics.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("phase") != "val":
            continue
        f1s = row.get("f1", "").strip()
        if not f1s:
            continue
        v = float(f1s)
        if row.get("epoch") == "0":
            ep0 = v
        best = v if best is None else max(best, v)

print(f"ep0_val_f1={ep0} best_val_f1={best}")
if ep0 is None or ep0 <= 0.35:
    print(f"[FAIL] ep0 F1 {ep0} <= 0.35")
    sys.exit(1)
if best is None or best <= 0.45:
    print(f"[FAIL] best F1 {best} <= 0.45")
    sys.exit(1)
print("[OK] MELD V rerun thresholds passed")
PY

bash scripts/start_sdavt_r4.sh report
echo "[OK] MELD V P4 rerun complete"

#!/usr/bin/env bash
# 重训完成后刷新 validate / audit / report / 快照 / TensorBoard。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"
STATUS="outputs_sdavt_v3_r4/status"
TS="$(date +%Y%m%d)"

"$PY" scripts/validate_p4_job_metrics.py --dataset meld R4_A_M_V || true
"$PY" scripts/audit_r4_training_health.py --queue-only --strict
"$PY" scripts/build_sdavt_r4_tables.py
"$PY" scripts/build_sdavt_r4_report.py

SNAP="$STATUS/r4_closeout_snapshot_${TS}.json"
"$PY" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
snap = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "queue_audit": "queue-only strict",
    "note": "post C4_C3 + MELD V sequential retrain",
}
Path("$SNAP").write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(f"Wrote {snap}")
PY

tmux kill-session -t sdavt_r4_tensorboard 2>/dev/null || true
tmux new-session -d -s sdavt_r4_tensorboard \
  "cd '$PROJECT_DIR' && bash scripts/tensorboard_sdavt_r4.sh 6008"

echo "[OK] refresh complete; TensorBoard http://127.0.0.1:6008 logdir=logs_sdavt_v3_r4/"

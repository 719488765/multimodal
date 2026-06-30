#!/usr/bin/env bash
# P3 失败 job 重跑完成后 → 验收 → 追加 P4 CREMA/MOSEI → 重启 worker
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
LOG="$PROJECT_DIR/outputs_sdavt_v3_r4/status/continue_after_p3_fix.log"
INTERVAL="${INTERVAL:-180}"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

_p3_pending_or_running() {
  python3 - <<'PY'
import json
from pathlib import Path
q = json.loads(Path("outputs_sdavt_v3_r4/experiment_queue.json").read_text())
ids = {"M3_M1_roberta", "M3_M7_combo"}
active = False
for j in q["jobs"]:
    if j.get("phase") == "p3_m3" and j["id"] in ids:
        if j.get("status") in ("pending", "running"):
            active = True
            print(j["id"], j["status"])
if not active:
    print("DONE")
PY
}

log "continue_r4_after_p3_fix watcher started"

while true; do
  if pgrep -f 'train.py.*M3_M1_roberta' >/dev/null 2>&1 || \
     pgrep -f 'train.py.*M3_M7_combo' >/dev/null 2>&1; then
    log "P3 rerun training active"
    sleep "$INTERVAL"
    continue
  fi

  state="$(_p3_pending_or_running | tail -1)"
  if [[ "$state" != "DONE" ]]; then
    log "waiting P3 rerun jobs: $state"
    sleep "$INTERVAL"
    continue
  fi

  log "P3 rerun finished — acceptance + P4 CREMA/MOSEI enqueue"
  python3 scripts/accept_sdavt_r4_p3_tier2.py --refresh-report >> "$LOG" 2>&1 || true
  bash scripts/start_sdavt_r4.sh gen p4_crema_mosei >> "$LOG" 2>&1 || true
  bash scripts/start_sdavt_r4.sh run p4_crema_mosei >> "$LOG" 2>&1 || true
  log "P4 CREMA/MOSEI workers launched"
  exit 0
done

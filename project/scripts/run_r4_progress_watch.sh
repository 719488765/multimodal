#!/usr/bin/env bash
# Lightweight R4 training progress logger.
#
# DISABLED BY DEFAULT: continuous nvidia-smi / queue polling is not needed after
# R4 close-out, and monitoring loops that spawn process scanners (ps/fuser) have
# previously fork-stormed this server.
#
# Opt-in only:
#   ENABLE_R4_PROGRESS_WATCH=1 bash scripts/run_r4_progress_watch.sh
set -euo pipefail

if [[ "${ENABLE_R4_PROGRESS_WATCH:-0}" != "1" ]]; then
  echo "[r4_progress_watch] DISABLED (default). Set ENABLE_R4_PROGRESS_WATCH=1 to run."
  echo "[r4_progress_watch] Prefer one-shot: python scripts/monitor_sdavt_r4.py --once"
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
OUT="outputs_sdavt_v3_r4/status/r4_progress_watch.log"
AVT="logs_sdavt_v3_r4/SDAVT_R4_R4_A_M_AVT/metrics.csv"
CREMA="logs_sdavt_v3_r4/SDAVT_R4_C4_C1_combo_acc/metrics.csv"
INTERVAL="${R4_PROGRESS_INTERVAL_SEC:-300}"

_log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

_last_val() {
  local file="$1" col="$2"
  [[ -f "$file" ]] || { echo "-"; return; }
  awk -F, -v c="$col" '
    NR>1 && $2=="val" { v=$(NF); if(c=="acc") v=$(NF-3); last=v }
    END { if(last=="") print "-"; else print last }
  ' "$file"
}

_meld_pending() {
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/scripts/r4_env.sh"
  "$R4_PYTHON" - <<'PY'
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
print(sum(1 for j in q["jobs"] if j.get("phase")=="p4_modal" and j.get("dataset")=="meld" and j.get("status") in ("pending","running")))
PY
}

_log "r4_progress_watch started (interval=${INTERVAL}s) ENABLE_R4_PROGRESS_WATCH=1"
while true; do
  avt_n=$(wc -l < "$AVT" 2>/dev/null || echo 0)
  crema_n=$(wc -l < "$CREMA" 2>/dev/null || echo 0)
  avt_f1=$(_last_val "$AVT" f1)
  crema_acc=$(_last_val "$CREMA" acc)
  # Prefer reading metrics files only; nvidia-smi optional and timeout-guarded.
  gpu=$(timeout 3 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '/' | sed 's/ %//g' || echo "?")
  meld=$(_meld_pending)
  _log "MELD_pending=$meld | AVT_lines=$avt_n val_f1=$avt_f1 | CREMA_lines=$crema_n val_acc=$crema_acc | GPU=${gpu:-?}"
  sleep "$INTERVAL"
done

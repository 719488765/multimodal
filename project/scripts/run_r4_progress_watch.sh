#!/usr/bin/env bash
# Lightweight R4 training progress logger (poll every 120s).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
OUT="outputs_sdavt_v3_r4/status/r4_progress_watch.log"
AVT="logs_sdavt_v3_r4/SDAVT_R4_R4_A_M_AVT/metrics.csv"
CREMA="logs_sdavt_v3_r4/SDAVT_R4_C4_C1_combo_acc/metrics.csv"

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
  source "$PROJECT_DIR/scripts/r4_env.sh"
  "$R4_PYTHON" - <<'PY'
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
print(sum(1 for j in q["jobs"] if j.get("phase")=="p4_modal" and j.get("dataset")=="meld" and j.get("status") in ("pending","running")))
PY
}

_log "r4_progress_watch started (interval=120s)"
while true; do
  avt_n=$(wc -l < "$AVT" 2>/dev/null || echo 0)
  crema_n=$(wc -l < "$CREMA" 2>/dev/null || echo 0)
  avt_f1=$(_last_val "$AVT" f1)
  crema_acc=$(_last_val "$CREMA" acc)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '/' | sed 's/ %//g')
  meld=$(_meld_pending)
  _log "MELD_pending=$meld | AVT_lines=$avt_n val_f1=$avt_f1 | CREMA_lines=$crema_n val_acc=$crema_acc | GPU=${gpu:-?}"
  sleep 120
done

#!/usr/bin/env bash
# Post-fix R4 pipeline: F_C_ES retrain after CREMA P4, MOSEI marker, then MELD unblocks via worker.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
QUEUE="outputs_sdavt_v3_r4/experiment_queue.json"
STATUS_DIR="outputs_sdavt_v3_r4/status"
LOG="$STATUS_DIR/post_fix_pipeline.log"
ENV_NAME="${ENV_NAME:-myenv310}"

_activate_conda() {
  for _c in \
    "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [[ -f "$_c" ]] && source "$_c" && conda activate "$ENV_NAME" && return 0
  done
  return 1
}

_pending_count() {
  local dataset="$1"
  python3 - "$QUEUE" "$dataset" <<'PY'
import json, sys
q = json.loads(open(sys.argv[1], encoding="utf-8").read())
ds = sys.argv[2]
n = sum(
    1 for j in q["jobs"]
    if j.get("phase") == "p4_modal" and j.get("dataset") == ds and j.get("status") == "pending"
)
print(n)
PY
}

_log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

_wait_crema_p4() {
  _log "waiting for CREMA P4 queue to drain..."
  while [[ "$(_pending_count crema)" -gt 0 ]]; do
    running="$(python3 - <<'PY'
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
print(any(j.get("status")=="running" and j.get("dataset")=="crema" for j in q["jobs"]))
PY
)"
    _log "CREMA pending=$(_pending_count crema) running=$running"
    sleep 120
  done
  _log "CREMA P4 complete"
}

_run_f_c_es() {
  if [[ -f "$STATUS_DIR/f_c_es_isolated_retrain_done" ]]; then
    _log "F_C_ES isolated retrain already done"
    return 0
  fi
  if [[ -f "$STATUS_DIR/f_c_es_antiof_retrain_done" ]]; then
    _log "F_C_ES prior retrain marker exists but isolated retrain pending — running isolated script"
  fi
  _log "starting F_C_ES isolated retrain on GPU1"
  export CUDA_VISIBLE_DEVICES=1
  bash scripts/run_f_c_es_isolated_retrain.sh >> "$STATUS_DIR/retrain_crema_es.log" 2>&1
  _log "F_C_ES isolated retrain done"
}

_start_mosei_worker() {
  if tmux has-session -t sdavt_r4_worker_gpu0 2>/dev/null; then
    _log "GPU0 worker already running"
    return 0
  fi
  _log "starting GPU0 worker for MOSEI P4 (after F_C_ES marker)"
  tmux new-session -d -s sdavt_r4_worker_gpu0
  tmux send-keys -t sdavt_r4_worker_gpu0 "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t sdavt_r4_worker_gpu0 "bash scripts/sdavt_r4_worker.sh 0" C-m
}

_wait_mosei_p4() {
  _log "waiting for MOSEI P4 queue to drain..."
  while [[ "$(_pending_count mosei)" -gt 0 ]]; do
    sleep 120
  done
  if python3 scripts/validate_p4_job_metrics.py --dataset mosei --strict; then
    touch "$STATUS_DIR/mosei_p4_retrain_done"
    _log "MOSEI P4 complete; strict validation passed; marker created"
  else
    _log "MOSEI P4 queue drained but strict validation FAILED — marker NOT created"
  fi
}

_wait_meld_p4() {
  _log "waiting for MELD P4 queue to drain..."
  while [[ "$(_pending_count meld)" -gt 0 ]]; do
    sleep 120
  done
  _log "MELD P4 complete"
}

_run_mosei_audio_rerun() {
  if [[ -f "$STATUS_DIR/mosei_audio_p4_rerun_done" ]]; then
    _log "MOSEI audio P4 rerun already done"
    return 0
  fi
  _log "starting MOSEI audio collapse rerun (AT/AVT/AV/A)"
  tmux kill-session -t sdavt_r4_worker_gpu0 2>/dev/null || true
  bash scripts/run_r4_mosei_audio_rerun.sh >> "$STATUS_DIR/mosei_audio_rerun.log" 2>&1
  touch "$STATUS_DIR/mosei_audio_p4_rerun_done"
  _log "MOSEI audio P4 rerun done"
}

_refresh_report() {
  bash scripts/start_sdavt_r4.sh report >> "$LOG" 2>&1 || true
}

main() {
  mkdir -p "$STATUS_DIR"
  _activate_conda || true
  _wait_crema_p4
  _run_f_c_es
  _start_mosei_worker
  _wait_mosei_p4
  _start_mosei_worker
  _wait_meld_p4
  _run_mosei_audio_rerun
  _refresh_report
  _log "pipeline complete; all post-fix retrain stages done"
}

main "$@"

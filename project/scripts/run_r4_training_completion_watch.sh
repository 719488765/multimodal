#!/usr/bin/env bash
# Chain post-audit: CREMA P3-C+ (GPU1) || MELD drain -> MOSEI rerun (GPU0) -> final audit.
# DISABLED BY DEFAULT after R4 close-out. Opt-in:
#   ENABLE_R4_WATCH=1 bash scripts/run_r4_training_completion_watch.sh
set -euo pipefail

if [[ "${ENABLE_R4_WATCH:-0}" != "1" ]]; then
  echo "[training_completion_watch] DISABLED (default). Set ENABLE_R4_WATCH=1 to run."
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"
STATUS="outputs_sdavt_v3_r4/status"
LOG="$STATUS/training_completion_watch.log"

_log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

_wait_tmux_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
}

_meld_pending() {
  "$PY" - <<'PY'
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
print(sum(1 for j in q["jobs"] if j.get("phase")=="p4_modal" and j.get("dataset")=="meld" and j.get("status") in ("pending","running")))
PY
}

_start_crema_p3c_plus() {
  if tmux has-session -t r4_crema_p3c_plus 2>/dev/null; then
    return 0
  fi
  if [[ -f logs_sdavt_v3_r4/SDAVT_R4_C4_C1_combo_acc/metrics.csv ]]; then
    _log "CREMA P3-C+ metrics exist — skip restart (Tier-2 review only)"
    return 0
  fi
  _log "starting CREMA P3-C+ on GPU1 (background)"
  tmux new-session -d -s r4_crema_p3c_plus \
    "cd \"$PROJECT_DIR\" && source scripts/r4_env.sh && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 && exec \"\$R4_PYTHON\" scripts/train.py --config config/sdavt_v3_r4/p3_c_plus/crema/C4_C1_combo_acc.yaml --mode pretrain --replace_log_dir >> $STATUS/crema_p3c_plus_live.log 2>&1"
}

_wait_meld_p4() {
  _log "waiting for MELD P4 queue to drain..."
  while [[ "$(_meld_pending)" -gt 0 ]]; do
    _log "MELD pending/running=$(_meld_pending)"
    sleep 120
  done
  if ! "$PY" scripts/validate_p4_job_metrics.py --dataset meld --strict; then
    _log "MELD strict validation FAILED — resetting failed jobs and rerunning"
    for job in $( "$PY" <<'PY'
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
print(" ".join(j["id"] for j in q["jobs"] if j.get("phase")=="p4_modal" and j.get("dataset")=="meld" and j.get("status")=="failed"))
PY
); do
      [[ -n "$job" ]] && "$PY" scripts/reset_r4_queue_jobs.py "$job" --phase p4_modal
    done
    tmux kill-session -t sdavt_r4_worker_gpu0 2>/dev/null || true
    tmux new-session -d -s sdavt_r4_worker_gpu0 \
      "cd \"$PROJECT_DIR\" && source scripts/r4_env.sh && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 && bash scripts/sdavt_r4_worker.sh 0 >> $STATUS/meld_p4_rerun.log 2>&1"
    while [[ "$(_meld_pending)" -gt 0 ]]; do sleep 120; done
    "$PY" scripts/validate_p4_job_metrics.py --dataset meld --strict || _log "MELD still failing after rerun"
  fi
}

_log "training completion watch started"

if [[ ! -f "$STATUS/f_c_es_isolated_retrain_done" ]]; then
  _log "F_C_ES marker missing — running restore from backup"
  "$PY" scripts/restore_f_c_es_from_backup.py >> "$LOG" 2>&1 || true
fi

_start_crema_p3c_plus
_wait_meld_p4

if [[ ! -f "$STATUS/mosei_audio_p4_rerun_done" ]]; then
  if ! tmux has-session -t r4_mosei_audio_rerun 2>/dev/null; then
    _log "starting MOSEI audio P4 rerun on GPU0"
    tmux kill-session -t sdavt_r4_worker_gpu0 2>/dev/null || true
    tmux new-session -d -s r4_mosei_audio_rerun \
      "cd \"$PROJECT_DIR\" && source scripts/r4_env.sh && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 && bash scripts/run_r4_mosei_audio_rerun.sh >> $STATUS/mosei_audio_rerun_live.log 2>&1"
  else
    _log "MOSEI audio rerun session already active — waiting"
  fi
  _wait_tmux_done r4_mosei_audio_rerun || true
  if "$PY" scripts/validate_p4_job_metrics.py --dataset mosei --strict; then
    touch "$STATUS/mosei_audio_p4_rerun_done"
    _log "MOSEI audio P4 rerun strict validation passed"
  else
    _log "MOSEI audio P4 rerun FAILED strict validation — marker NOT set"
  fi
fi

if [[ ! -f "$STATUS/crema_p3c_plus_done" ]]; then
  _log "waiting for CREMA P3-C+ to finish (GPU1 background)"
  _wait_tmux_done r4_crema_p3c_plus || true
  if "$PY" <<'PY'
import csv
from pathlib import Path
p=Path("logs_sdavt_v3_r4/SDAVT_R4_C4_C1_combo_acc/metrics.csv")
best=None
if p.is_file():
    for r in csv.DictReader(p.open()):
        if r.get("phase")=="val" and r.get("accuracy"):
            v=float(r["accuracy"]); best=v if best is None else max(best,v)
if best is None or best < 0.63:
    raise SystemExit(1)
print(f"best_acc={best}")
PY
  then
    touch "$STATUS/crema_p3c_plus_done"
    _log "CREMA P3-C+ Tier-2 Acc target met"
  else
    _log "CREMA P3-C+ finished but Acc < 0.63 — review logs"
  fi
fi

"$PY" scripts/audit_r4_training_health.py || true
"$PY" scripts/build_sdavt_r4_tables.py || true
"$PY" scripts/build_sdavt_r4_report.py
_log "training completion watch done"

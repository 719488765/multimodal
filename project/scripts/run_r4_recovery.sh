#!/usr/bin/env bash
# Recovery: unstick pipeline, finish MELD AVT, MOSEI audio x4, CREMA P3-C+.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
STATUS="outputs_sdavt_v3_r4/status"
LOG="$STATUS/recovery_$(date +%Y%m%d_%H%M%S).log"

_log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

_log "=== R4 recovery start ==="

# 1. Kill zombie tmux (avoid broad pkill that may hang)
tmux kill-session -t r4_crema_p3c_plus 2>/dev/null || true
tmux kill-session -t r4_training_completion_watch 2>/dev/null || true
tmux kill-session -t r4_mosei_audio_watch 2>/dev/null || true
tmux kill-session -t sdavt_r4_worker_gpu0 2>/dev/null || true
sleep 1

# 2. Fix misleading MOSEI marker (4 jobs still pending)
rm -f "$STATUS/mosei_p4_retrain_done"

# 3. AT already valid — mark done without rerun
if python3 scripts/validate_p4_job_metrics.py R4_A_M_AT --dataset meld >/dev/null 2>&1; then
  python3 scripts/mark_r4_queue_job_done.py R4_A_M_AT --run-dir SDAVT_R4_R4_A_M_AT
  _log "R4_A_M_AT marked done (F1=0.674 valid)"
fi

# 4. Reset zombie AVT — archive empty slot
ARCH="outputs_sdavt_v3_r4/archived/p4_meld_avt_zombie_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH"
for base in logs_sdavt_v3_r4 checkpoints_sdavt_v3_r4; do
  d="$base/SDAVT_R4_R4_A_M_AVT"
  if [[ -d "$d" ]]; then
    rm -rf "$ARCH/$(basename "$d")"
    mv "$d" "$ARCH/" && _log "archived $d"
  fi
done
python3 scripts/reset_r4_queue_jobs.py R4_A_M_AVT --phase p4_modal

# 5. Archive empty CREMA P3-C+ slot
ARCH_C="outputs_sdavt_v3_r4/archived/p3c_plus_stuck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCH_C"
for base in logs_sdavt_v3_r4 checkpoints_sdavt_v3_r4; do
  d="$base/SDAVT_R4_C4_C1_combo_acc"
  if [[ -d "$d" ]]; then
    rm -rf "$ARCH_C/$(basename "$d")"
    mv "$d" "$ARCH_C/" && _log "archived $d"
  fi
done
rm -f "$STATUS/crema_p3c_plus_done"

# 6. Start GPU0 worker (MELD AVT only)
_log "starting GPU0 worker for MELD AVT (myenv310 python)"
tmux new-session -d -s sdavt_r4_worker_gpu0 \
  "cd \"$PROJECT_DIR\" && source scripts/r4_env.sh && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 && exec bash scripts/sdavt_r4_worker.sh 0 >> $STATUS/meld_worker_recovery.log 2>&1"

# 7. Start CREMA P3-C+ on GPU1 (parallel)
_log "starting CREMA P3-C+ on GPU1 (myenv310 python)"
tmux new-session -d -s r4_crema_p3c_plus \
  "cd \"$PROJECT_DIR\" && source scripts/r4_env.sh && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 && exec \"\$R4_PYTHON\" scripts/train.py --config config/sdavt_v3_r4/p3_c_plus/crema/C4_C1_combo_acc.yaml --mode pretrain --replace_log_dir >> $STATUS/crema_p3c_plus_live.log 2>&1"

# 8. Restart completion watch (MELD drain -> MOSEI -> wait CREMA)
_log "starting completion watch"
tmux new-session -d -s r4_training_completion_watch \
  "cd \"$PROJECT_DIR\" && exec bash scripts/run_r4_training_completion_watch.sh >> $STATUS/training_completion_watch.log 2>&1"

_log "=== recovery launched ==="
tmux ls 2>/dev/null || true

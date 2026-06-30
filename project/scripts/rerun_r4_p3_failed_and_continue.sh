#!/usr/bin/env bash
# 修复 tokenizer 后重跑 P3 失败项，并启动后续 P4 CREMA/MOSEI 监视器
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

ENV_NAME="${ENV_NAME:-myenv310}"
CONDA_SH=""
for _c in \
  "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  [[ -f "$_c" ]] && CONDA_SH="$_c" && break
done
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _b="$(conda info --base 2>/dev/null || true)"
  [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
fi
[[ -n "$CONDA_SH" ]] || { echo "[ERROR] conda.sh not found"; exit 1; }

echo "[1/4] Reset failed P3 jobs -> pending"
python3 scripts/reset_r4_queue_jobs.py M3_M1_roberta M3_M7_combo --phase p3_m3

echo "[2/4] Launch R4 workers (no queue regen)"
for s in sdavt_r4_worker_gpu0 sdavt_r4_worker_gpu1 sdavt_r4_tb continue_r4_after_p3_fix; do
  tmux kill-session -t "$s" 2>/dev/null || true
done
tmux new-session -d -s sdavt_r4_tb "bash $PROJECT_DIR/scripts/tensorboard_sdavt_r4.sh 6008"
for GPU in 0 1; do
  tmux new-session -d -s "sdavt_r4_worker_gpu${GPU}"
  tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "export CUDA_VISIBLE_DEVICES=$GPU" C-m
  tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "bash scripts/sdavt_r4_worker.sh $GPU" C-m
done

echo "[3/4] Start post-P3 watcher (accept + P4 CREMA/MOSEI)"
tmux new-session -d -s continue_r4_after_p3_fix "bash $PROJECT_DIR/scripts/continue_r4_after_p3_fix.sh"

echo "[4/4] P5 deploy preset already points to M3_M3_uniform winner"
echo "  To switch Agent: bash scripts/apply_deploy_preset.sh sdavt_meld_v3_r4"
echo "Monitor: bash scripts/start_sdavt_r4.sh status"
echo "Log: outputs_sdavt_v3_r4/status/continue_after_p3_fix.log"

#!/usr/bin/env bash
# SDAVT v3 Phase 1 单域基线 — tmux 启动（日志写入 logs_sdavt_v3/）
#
# 用法:
#   ./scripts/start_sdavt_v3_p1_tmux.sh p1          # GPU0 MELD S1-M1 + GPU1 CREMA S1-C0
#   ./scripts/start_sdavt_v3_p1_tmux.sh p1mosei     # GPU0 MELD S1-M1 + GPU1 MOSEI S1-O0
#   ./scripts/start_sdavt_v3_p1_tmux.sh meld        # 仅 MELD S1-M1（GPU0）
#   ./scripts/start_sdavt_v3_p1_tmux.sh m0          # 仅 MELD S1-M0 AP1 对齐（GPU0）
#   ./scripts/start_sdavt_v3_p1_tmux.sh crema       # 仅 CREMA S1-C0（GPU1）
#   ./scripts/start_sdavt_v3_p1_tmux.sh mosei      # 仅 MOSEI S1-O0（GPU1）
#   ./scripts/start_sdavt_v3_p1_tmux.sh list       # 打印命令
#
# 环境变量: PROJECT_DIR, ENV_NAME, CONDA_BASE, GPU_MELD (默认0), GPU_OTHER (默认1)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-myenv310}"
GPU_MELD="${GPU_MELD:-0}"
GPU_OTHER="${GPU_OTHER:-1}"
PHASE="${1:-p1}"

CFG="config/sdavt_v3"
SKIP_AUDIO_EXTRACT="${SKIP_AUDIO_EXTRACT:-1}"

CONDA_SH=""
if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
else
  for _c in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/usr/local/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$_c" ]]; then
      CONDA_SH="$_c"
      break
    fi
  done
fi
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$_base" && -f "$_base/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$_base/etc/profile.d/conda.sh"
  fi
fi

CMD_M1="SKIP_AUDIO_EXTRACT=${SKIP_AUDIO_EXTRACT} bash scripts/train_sdavt_meld_v2.sh"
CMD_M0="python3 scripts/train.py --config ${CFG}/meld/S1_M0_AVT_ES_baseline.yaml --mode pretrain"
CMD_CREMA="python3 scripts/train.py --config ${CFG}/crema/S1_C0_AVT_ES_baseline.yaml --mode pretrain"
CMD_MOSEI="python3 scripts/train.py --config ${CFG}/mosei/S1_O0_AVT_ES_npy.yaml --mode pretrain"

# 格式: session:GPU_ID:command
declare -a RUNS=()

pick_runs() {
  case "$PHASE" in
    p1)
      RUNS=(
        "sdavt_m1_meld:${GPU_MELD}:${CMD_M1}"
        "sdavt_c0_crema:${GPU_OTHER}:${CMD_CREMA}"
      )
      ;;
    p1mosei)
      RUNS=(
        "sdavt_m1_meld:${GPU_MELD}:${CMD_M1}"
        "sdavt_o0_mosei:${GPU_OTHER}:${CMD_MOSEI}"
      )
      ;;
    meld)
      RUNS=("sdavt_m1_meld:${GPU_MELD}:${CMD_M1}")
      ;;
    m0)
      RUNS=("sdavt_m0_meld:${GPU_MELD}:${CMD_M0}")
      ;;
    crema)
      RUNS=("sdavt_c0_crema:${GPU_OTHER}:${CMD_CREMA}")
      ;;
    mosei)
      RUNS=("sdavt_o0_mosei:${GPU_OTHER}:${CMD_MOSEI}")
      ;;
    list)
      echo "# p1: MELD S1-M1 (GPU${GPU_MELD}) + CREMA S1-C0 (GPU${GPU_OTHER})"
      echo "# p1mosei: MELD S1-M1 (GPU${GPU_MELD}) + MOSEI S1-O0 (GPU${GPU_OTHER})"
      echo "SKIP_AUDIO_EXTRACT=1 $CMD_M1"
      echo "$CMD_M0"
      echo "$CMD_CREMA"
      echo "$CMD_MOSEI"
      return 0
      ;;
    *)
      echo "Usage: $0 {p1|p1mosei|meld|m0|crema|mosei|list}"
      exit 1
      ;;
  esac
}

if [[ "$PHASE" == "list" ]]; then
  pick_runs
  exit 0
fi

pick_runs

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] PROJECT_DIR not found: $PROJECT_DIR"
  exit 1
fi

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh not found. Set CONDA_BASE or install conda."
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs_sdavt_v3" "$PROJECT_DIR/checkpoints_sdavt_v3" "$PROJECT_DIR/outputs_sdavt_v3"

for item in "${RUNS[@]}"; do
  session="${item%%:*}"
  rest="${item#*:}"
  gpu="${rest%%:*}"
  cmd="${rest#*:}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[INFO] session exists, skip: $session"
    continue
  fi

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
  tmux send-keys -t "$session" "$cmd" C-m
  echo "[OK] tmux -t $session  GPU=$gpu"
  echo "     $cmd"
done

echo
echo "TensorBoard: bash scripts/tensorboard_sdavt_v3.sh 6007"
echo "会话列表:    tmux ls"
echo "附着 MELD:   tmux attach -t sdavt_m1_meld"
echo "汇总结果:    python3 scripts/summarize_sdavt_v3_results.py"

#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 分批节流启动重跑实验（按批次并发，支持 auto/single/dual 与 dry-run）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="myenv310"
USE_CONDA="${USE_CONDA:-1}" # 1 | 0

resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="$base/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/opt/anaconda3/etc/profile.d/conda.sh"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      CONDA_SH="$c"
      return 0
    fi
  done
  return 1
}

GROUP="${1:-all}"        # mainline | expansion | extra | all
MODE="${2:-auto}"        # auto | single | dual
DRY_RUN="${3:-0}"        # 0 | 1
BATCH_SIZE="${4:-2}"     # 每批最多启动的会话数
POLL_SEC="${5:-30}"      # 批次等待轮询间隔（秒）
MAX_RETRY="${6:-0}"      # 兼容保留：当前脚本不自动重试
RETRY_WAIT_SEC="${7:-20}" # 兼容保留：当前脚本不自动重试

GPU_SINGLE="${GPU_SINGLE:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
MIN_RUN_FREE_MB="${MIN_RUN_FREE_MB:-3000}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] project dir not found: $PROJECT_DIR"
  exit 1
fi

if [[ "$USE_CONDA" != "0" && "$DRY_RUN" != "1" ]]; then
  if ! resolve_conda_sh; then
    echo "[ERROR] conda script not found. Tried CONDA_SH and common locations."
    echo "        You can set env manually: export CONDA_SH=/path/to/conda.sh"
    exit 1
  fi
fi

if [[ "$MODE" != "auto" && "$MODE" != "single" && "$MODE" != "dual" ]]; then
  echo "[ERROR] unsupported mode: $MODE (use auto|single|dual)"
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[ERROR] unsupported dry-run flag: $DRY_RUN (use 0|1)"
  exit 1
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] BATCH_SIZE must be positive integer"
  exit 1
fi
if ! [[ "$POLL_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] POLL_SEC must be positive integer"
  exit 1
fi
if ! [[ "$MAX_RETRY" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] MAX_RETRY must be non-negative integer"
  exit 1
fi
if ! [[ "$RETRY_WAIT_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] RETRY_WAIT_SEC must be positive integer"
  exit 1
fi

free0=0
free1=0
if command -v nvidia-smi >/dev/null 2>&1; then
  free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
  free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
fi

if [[ "$MODE" == "auto" ]]; then
  if [[ "${free0:-0}" -ge "$MIN_FREE_MB" && "${free1:-0}" -ge "$MIN_FREE_MB" ]]; then
    MODE="dual"
  else
    MODE="single"
  fi
fi

STATUS_DIR="$PROJECT_DIR/logs_rerun/.launcher_status"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$STATUS_DIR"
fi
RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAILED_SESSIONS_FILE="$STATUS_DIR/failed_sessions_${RUN_ID}.txt"
FAILED_ITEMS_FILE="$STATUS_DIR/failed_items_${RUN_ID}.txt"
LATEST_FAILED_SESSIONS_FILE="$STATUS_DIR/failed_sessions_latest.txt"
LATEST_FAILED_ITEMS_FILE="$STATUS_DIR/failed_items_latest.txt"
declare -a GLOBAL_FAILED_ITEMS=()

echo "[INFO] group=$GROUP mode=$MODE dry_run=$DRY_RUN batch_size=$BATCH_SIZE poll_sec=$POLL_SEC"
echo "[INFO] retry policy=disabled (manual retry by user)"
echo "[INFO] use_conda=$USE_CONDA conda_sh=$CONDA_SH"
echo "[INFO] free_mem_mb(gpu0,gpu1)=(${free0:-NA},${free1:-NA})"
echo "[INFO] min_run_free_mb=$MIN_RUN_FREE_MB (per launch gate)"

build_items() {
  local group="$1"
  local mode="$2"
  local -n out_ref="$3"

  # 条目格式: session:gpu:cmd
  if [[ "$group" == "mainline" || "$group" == "all" ]]; then
    if [[ "$mode" == "single" ]]; then
      out_ref+=(
        "rerun_at_noda:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AT_noDA.yaml --mode pretrain"
        "rerun_at_da:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AT_DA.yaml --mode pretrain"
        "rerun_vt_noda:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_VT_noDA.yaml --mode pretrain"
        "rerun_avt_noda:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain"
        "rerun_avt_da:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain"
        "rerun_avt_es:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_emotion_shift.yaml --mode pretrain"
      )
    else
      out_ref+=(
        "rerun_at_noda:${GPU0}:python3 scripts/train.py --config config/rerun/config_AT_noDA.yaml --mode pretrain"
        "rerun_at_da:${GPU1}:python3 scripts/train.py --config config/rerun/config_AT_DA.yaml --mode pretrain"
        "rerun_vt_noda:${GPU0}:python3 scripts/train.py --config config/rerun/config_VT_noDA.yaml --mode pretrain"
        "rerun_avt_noda:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain"
        "rerun_avt_da:${GPU0}:python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain"
        "rerun_avt_es:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_emotion_shift.yaml --mode pretrain"
      )
    fi
  fi

  if [[ "$group" == "expansion" || "$group" == "all" ]]; then
    if [[ "$mode" == "single" ]]; then
      out_ref+=(
        "rerun_t_only:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_text_only.yaml --mode pretrain"
        "rerun_a_only:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_audio_only.yaml --mode pretrain"
        "rerun_v_only:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_video_only.yaml --mode pretrain"
        "rerun_avt_da_w002:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w002.yaml --mode pretrain"
        "rerun_avt_da_w005:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w005.yaml --mode pretrain"
        "rerun_avt_da_w010:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w010.yaml --mode pretrain"
      )
    else
      out_ref+=(
        "rerun_t_only:${GPU0}:python3 scripts/train.py --config config/rerun/config_text_only.yaml --mode pretrain"
        "rerun_a_only:${GPU1}:python3 scripts/train.py --config config/rerun/config_audio_only.yaml --mode pretrain"
        "rerun_v_only:${GPU0}:python3 scripts/train.py --config config/rerun/config_video_only.yaml --mode pretrain"
        "rerun_avt_da_w002:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w002.yaml --mode pretrain"
        "rerun_avt_da_w005:${GPU0}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w005.yaml --mode pretrain"
        "rerun_avt_da_w010:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w010.yaml --mode pretrain"
      )
    fi
  fi

  if [[ "$group" == "extra" || "$group" == "all" ]]; then
    if [[ "$mode" == "single" ]]; then
      out_ref+=(
        "rerun_avt_noda_uniform:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_uniform.yaml --mode pretrain"
        "rerun_avt_da_uniform:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA_uniform.yaml --mode pretrain"
        "rerun_avt_noda_lr5e5:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_lr5e5.yaml --mode pretrain"
        "rerun_avt_da_w005_lr5e5:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w005_lr5e5.yaml --mode pretrain"
        "rerun_avt_noda_seed3407:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_seed3407.yaml --mode pretrain"
        "rerun_avt_da_seed3407:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA_seed3407.yaml --mode pretrain"
      )
    else
      out_ref+=(
        "rerun_avt_noda_uniform:${GPU0}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_uniform.yaml --mode pretrain"
        "rerun_avt_da_uniform:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_DA_uniform.yaml --mode pretrain"
        "rerun_avt_noda_lr5e5:${GPU0}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_lr5e5.yaml --mode pretrain"
        "rerun_avt_da_w005_lr5e5:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_DA_w005_lr5e5.yaml --mode pretrain"
        "rerun_avt_noda_seed3407:${GPU0}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_seed3407.yaml --mode pretrain"
        "rerun_avt_da_seed3407:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_DA_seed3407.yaml --mode pretrain"
      )
    fi
  fi
}

if [[ "$GROUP" != "mainline" && "$GROUP" != "expansion" && "$GROUP" != "extra" && "$GROUP" != "all" ]]; then
  echo "[ERROR] unsupported group: $GROUP (use mainline|expansion|extra|all)"
  exit 1
fi

declare -a ITEMS=()
build_items "$GROUP" "$MODE" ITEMS

if [[ "${#ITEMS[@]}" -eq 0 ]]; then
  echo "[ERROR] no run items generated"
  exit 1
fi

wait_batch_done() {
  local -a status_files=("$@")
  local pending
  while true; do
    pending=0
    for sf in "${status_files[@]}"; do
      if [[ ! -f "$sf" ]]; then
        pending=$((pending + 1))
      fi
    done
    if [[ "$pending" -eq 0 ]]; then
      break
    fi
    echo "[WAIT] batch pending tasks=$pending, sleep ${POLL_SEC}s..."
    sleep "$POLL_SEC"
  done
}

launch_item() {
  local item="$1"
  local session rest gpu cmd status_file
  session="${item%%:*}"
  rest="${item#*:}"
  gpu="${rest%%:*}"
  cmd="${rest#*:}"
  status_file="$STATUS_DIR/${session}.attempt0.exitcode"

  # 幂等保护：若该任务已有完成状态且成功，默认跳过，避免重连后重复训练
  if [[ -f "$status_file" ]]; then
    existing_code="$(<"$status_file")"
    if [[ "$existing_code" == "0" && "${FORCE_RERUN:-0}" != "1" ]]; then
      echo "[INFO] skip completed task: $session (status_file=$status_file)"
      return 3
    fi
  fi

  rm -f "$status_file"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[INFO] session exists, skip: $session"
    return 2
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    while true; do
      cur_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((gpu + 1))p" | tr -d ' ')
      if [[ -n "$cur_free" && "$cur_free" =~ ^[0-9]+$ && "$cur_free" -ge "$MIN_RUN_FREE_MB" ]]; then
        break
      fi
      echo "[WAIT] GPU $gpu free=${cur_free:-NA}MB < ${MIN_RUN_FREE_MB}MB, sleep ${POLL_SEC}s..."
      sleep "$POLL_SEC"
    done
  fi

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  if [[ "$USE_CONDA" != "0" ]]; then
    tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
    tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  fi
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
  tmux send-keys -t "$session" "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" C-m
  tmux send-keys -t "$session" "$cmd; ec=\$?; echo \$ec > \"$status_file\"; echo \"[TRAIN_DONE] exit_code=\$ec\"" C-m
  echo "[OK] started $session on CUDA_VISIBLE_DEVICES=$gpu (no auto-retry, session kept)"
  echo "[INFO] session=$session status_file=$status_file"
  return 0
}

total="${#ITEMS[@]}"
idx=0
batch_no=1

while [[ "$idx" -lt "$total" ]]; do
  end=$((idx + BATCH_SIZE))
  if [[ "$end" -gt "$total" ]]; then
    end="$total"
  fi

  echo
  echo "=============================="
  echo "[INFO] batch #$batch_no items [$((idx + 1))..$end]/$total"
  echo "=============================="

  declare -a BATCH_ITEMS=()
  declare -a STARTED_STATUS_FILES=()
  i="$idx"
  while [[ "$i" -lt "$end" ]]; do
    item="${ITEMS[$i]}"
    session="${item%%:*}"
    rest="${item#*:}"
    gpu="${rest%%:*}"
    cmd="${rest#*:}"

    if [[ "$DRY_RUN" == "1" ]]; then
      if tmux has-session -t "$session" 2>/dev/null; then
        echo "[PLAN] skip existing session: $session"
      else
        echo "[PLAN] start $session on CUDA_VISIBLE_DEVICES=$gpu"
        echo "       cmd: $cmd"
      fi
      i=$((i + 1))
      continue
    fi

    BATCH_ITEMS+=("$item")

    if launch_item "$item"; then
      STARTED_STATUS_FILES+=("$STATUS_DIR/${session}.attempt0.exitcode")
    fi
    i=$((i + 1))
  done

  idx="$end"
  batch_no=$((batch_no + 1))

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  if [[ "${#STARTED_STATUS_FILES[@]}" -gt 0 ]]; then
    wait_batch_done "${STARTED_STATUS_FILES[@]}"
  fi

  declare -a FAILED_ITEMS=()
  for r_item in "${BATCH_ITEMS[@]}"; do
    r_session="${r_item%%:*}"
    r_status_file="$STATUS_DIR/${r_session}.attempt0.exitcode"
    if [[ ! -f "$r_status_file" ]]; then
      echo "[WARN] missing exitcode file: $r_status_file"
      FAILED_ITEMS+=("$r_item")
      continue
    fi
    r_code="$(<"$r_status_file")"
    if [[ "$r_code" != "0" ]]; then
      echo "[WARN] failed: $r_session exit_code=$r_code"
      FAILED_ITEMS+=("$r_item")
    fi
  done

  if [[ "${#FAILED_ITEMS[@]}" -gt 0 ]]; then
    echo "[WARN] batch has failures (auto-retry disabled):"
    for x in "${FAILED_ITEMS[@]}"; do
      x_session="${x%%:*}"
      echo "  - $x_session"
      GLOBAL_FAILED_ITEMS+=("$x")
    done
  fi
done

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DONE] dry-run finished. No tmux sessions were started."
else
  : > "$FAILED_SESSIONS_FILE"
  : > "$FAILED_ITEMS_FILE"
  for item in "${GLOBAL_FAILED_ITEMS[@]}"; do
    s="${item%%:*}"
    echo "$s" >> "$FAILED_SESSIONS_FILE"
    echo "$item" >> "$FAILED_ITEMS_FILE"
  done
  cp "$FAILED_SESSIONS_FILE" "$LATEST_FAILED_SESSIONS_FILE"
  cp "$FAILED_ITEMS_FILE" "$LATEST_FAILED_ITEMS_FILE"

  if [[ "${#GLOBAL_FAILED_ITEMS[@]}" -gt 0 ]]; then
    echo "[WARN] unrecovered failures exported:"
    echo "       sessions: $FAILED_SESSIONS_FILE"
    echo "       items   : $FAILED_ITEMS_FILE"
    echo "       latest  : $LATEST_FAILED_ITEMS_FILE"
  else
    echo "[INFO] no unrecovered failures. Empty list generated:"
    echo "       $FAILED_ITEMS_FILE"
  fi
  echo "[DONE] throttled launch completed. Check current sessions with: tmux ls"
fi

#!/usr/bin/env bash
# Author: AI
# Date: 2026-04-11
# Description: 双卡交替绑定 CUDA_VISIBLE_DEVICES，启动指定阶段的准确率序列实验（日志在 logs_accuracy_seq）
#
# 用法:
#   ./scripts/start_accuracy_seq_gpuaware.sh ap2 auto    # 根据显存自动选 single/dual
#   ./scripts/start_accuracy_seq_gpuaware.sh ap2 dual 0  # 强制双卡，DRY_RUN=1 仅打印计划
# 参数:
#   $1 阶段: ap0|ap1|ap2|ap2opt|ap3|ap4（勿用 all，避免一次性占满）
#   $2 模式: auto|single|dual
#   $3 DRY_RUN: 0|1
#
# 环境变量: GPU0 GPU1 GPU_SINGLE MIN_FREE_MB PROJECT_DIR ENV_NAME CONDA_BASE（同主线 gpuaware 脚本）

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-myenv310}"
PHASE="${1:?phase required}"
MODE="${2:-auto}"
DRY_RUN="${3:-0}"
GPU_SINGLE="${GPU_SINGLE:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"

CONDA_SH=""
if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
else
  for _c in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$_c" ]]; then CONDA_SH="$_c"; break; fi
  done
fi
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _b="$(conda info --base 2>/dev/null || true)"
  [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
fi

if [[ "$PHASE" == "all" ]]; then
  echo "[ERROR] 请勿对本脚本使用 all；请分阶段调用以降低 OOM 风险。"
  exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] PROJECT_DIR not found: $PROJECT_DIR"
  exit 1
fi

if [[ "$DRY_RUN" != "1" && ( -z "$CONDA_SH" || ! -f "$CONDA_SH" ) ]]; then
  echo "[ERROR] conda.sh not found"
  exit 1
fi

free0=0
free1=0
if command -v nvidia-smi >/dev/null 2>&1; then
  free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ' || echo 0)
  free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ' || echo 0)
fi

if [[ "$MODE" == "auto" ]]; then
  if [[ "${free0:-0}" -ge "$MIN_FREE_MB" && "${free1:-0}" -ge "$MIN_FREE_MB" ]]; then
    MODE="dual"
  else
    MODE="single"
  fi
fi

echo "[INFO] phase=$PHASE mode=$MODE dry_run=$DRY_RUN free_mb=($free0,$free1)"

CFG="config/rerun/accuracy_plan"

case "$PHASE" in
  ap0)
    RUNS=(
      "aseq_ap0_std:python3 scripts/train.py --config $CFG/ap0_AVT_noDA_standard_full50_s3407.yaml --mode pretrain"
    ) ;;
  ap1)
    RUNS=(
      "aseq_ap1_vt_cr:python3 scripts/train.py --config $CFG/ap1_VT_crema_only_s3407.yaml --mode pretrain"
      "aseq_ap1_vt_me:python3 scripts/train.py --config $CFG/ap1_VT_meld_only_s3407.yaml --mode pretrain"
      "aseq_ap1_vt_mo:python3 scripts/train.py --config $CFG/ap1_VT_mosei_only_s3407.yaml --mode pretrain"
      "aseq_ap1_avt_cr:python3 scripts/train.py --config $CFG/ap1_AVT_ES_crema_only_s3407.yaml --mode pretrain"
      "aseq_ap1_avt_me:python3 scripts/train.py --config $CFG/ap1_AVT_ES_meld_only_s3407.yaml --mode pretrain"
      "aseq_ap1_avt_mo:python3 scripts/train.py --config $CFG/ap1_AVT_ES_mosei_only_s3407.yaml --mode pretrain"
    ) ;;
  ap2)
    RUNS=(
      "aseq_ap2_base:python3 scripts/train.py --config $CFG/ap2_ES_baseline_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_m1:python3 scripts/train.py --config $CFG/ap2_M1_effbatch8_ES_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_m2:python3 scripts/train.py --config $CFG/ap2_M2_lr5e5_ES_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_m3:python3 scripts/train.py --config $CFG/ap2_M3_uniform_ES_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_m4a:python3 scripts/train.py --config $CFG/ap2_M4_plain_ce_ES_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_m4b:python3 scripts/train.py --config $CFG/ap2_M4_focal_ES_3ds_s3407.yaml --mode pretrain"
    ) ;;
  ap2opt)
    RUNS=(
      "aseq_ap2_fcb:python3 scripts/train.py --config $CFG/ap2_opt_fixed_cb_ES_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_lsm:python3 scripts/train.py --config $CFG/ap2_opt_label_smooth_ES_3ds_s3407.yaml --mode pretrain"
      "aseq_ap2_bblr:python3 scripts/train.py --config $CFG/ap2_opt_backbone_lr_ES_3ds_s3407.yaml --mode pretrain"
    ) ;;
  ap3)
    RUNS=(
      "aseq_ap3_std:python3 scripts/train.py --config $CFG/ap3_fusion_standard_3ds_s3407.yaml --mode pretrain"
      "aseq_ap3_lf_tx:python3 scripts/train.py --config $CFG/ap3_fusion_leader_text_3ds_s3407.yaml --mode pretrain"
      "aseq_ap3_lf_au:python3 scripts/train.py --config $CFG/ap3_fusion_leader_audio_3ds_s3407.yaml --mode pretrain"
      "aseq_ap3_ts:python3 scripts/train.py --config $CFG/ap3_fusion_two_stage_3ds_s3407.yaml --mode pretrain"
    ) ;;
  ap4)
    RUNS=(
      "aseq_ap4_da_def:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_accuracy_seq.yaml --mode pretrain"
      "aseq_ap4_da_w02:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w002_accuracy_seq.yaml --mode pretrain"
      "aseq_ap4_da_w05:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w005_accuracy_seq.yaml --mode pretrain"
      "aseq_ap4_da_w10:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w010_accuracy_seq.yaml --mode pretrain"
      "aseq_ap4_da_w05lr:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w005_lr5e5_accuracy_seq.yaml --mode pretrain"
      "aseq_ap4_da_uni:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_uniform_accuracy_seq.yaml --mode pretrain"
      "aseq_ap4_da_s34:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_seed3407_accuracy_seq.yaml --mode pretrain"
    ) ;;
  *)
    echo "[ERROR] unknown phase: $PHASE"
    exit 1
    ;;
esac

idx=0
for item in "${RUNS[@]}"; do
  session="${item%%:*}"
  cmd="${item#*:}"
  if [[ "$MODE" == "dual" ]]; then
    if (( idx % 2 == 0 )); then gpu="$GPU0"; else gpu="$GPU1"; fi
  else
    gpu="$GPU_SINGLE"
  fi
  idx=$((idx + 1))

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[PLAN] $session -> GPU $gpu :: $cmd"
    continue
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[INFO] skip existing: $session"
    continue
  fi

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
  tmux send-keys -t "$session" "$cmd" C-m
  echo "[OK] $session on GPU $gpu"
done

[[ "$DRY_RUN" == "1" ]] && echo "[INFO] dry-run done."

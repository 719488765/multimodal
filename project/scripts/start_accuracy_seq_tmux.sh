#!/usr/bin/env bash
# Author: AI
# Date: 2026-04-11
# Description: 按阶段启动「准确率优化实验序列」tmux 会话（日志写入 logs_accuracy_seq，与 logs_rerun 隔离）
#
# 用法:
#   ./scripts/start_accuracy_seq_tmux.sh ap0              # 仅阶段0（1 个实验）
#   ./scripts/start_accuracy_seq_tmux.sh ap1              # 单数据集上界（6 个）
#   ./scripts/start_accuracy_seq_tmux.sh ap2              # 配方消融（不含 opt）
#   ./scripts/start_accuracy_seq_tmux.sh ap2opt           # 可选配方（fixed_cb / label_smooth / backbone_lr）
#   ./scripts/start_accuracy_seq_tmux.sh ap3              # 融合消融
#   ./scripts/start_accuracy_seq_tmux.sh ap4              # 域适应（7 个，见 accuracy_plan/ap4_*_accuracy_seq.yaml）
#   ./scripts/start_accuracy_seq_tmux.sh all              # 上述全部（慎用：同卡多任务易 OOM）
#   ./scripts/start_accuracy_seq_tmux.sh list             # 仅打印命令
#
# 环境变量:
#   PROJECT_DIR   默认本仓库 project 根目录（自动推断）
#   ENV_NAME      conda 环境名，默认 myenv310
#   CONDA_BASE    可选，指向 conda 根目录（含 etc/profile.d/conda.sh）
#   GPU_ID        默认 0，写入 CUDA_VISIBLE_DEVICES

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-myenv310}"
GPU_ID="${GPU_ID:-0}"
PHASE="${1:-}"

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

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] PROJECT_DIR not found: $PROJECT_DIR"
  exit 1
fi

if [[ "$PHASE" != "list" && ( -z "$CONDA_SH" || ! -f "$CONDA_SH" ) ]]; then
  echo "[ERROR] conda.sh not found. Set CONDA_BASE or install conda."
  exit 1
fi

CFG="config/rerun/accuracy_plan"

declare -a RUNS_AP0=(
  "aseq_ap0_std:python3 scripts/train.py --config $CFG/ap0_AVT_noDA_standard_full50_s3407.yaml --mode pretrain"
)

declare -a RUNS_AP1=(
  "aseq_ap1_vt_cr:python3 scripts/train.py --config $CFG/ap1_VT_crema_only_s3407.yaml --mode pretrain"
  "aseq_ap1_vt_me:python3 scripts/train.py --config $CFG/ap1_VT_meld_only_s3407.yaml --mode pretrain"
  "aseq_ap1_vt_mo:python3 scripts/train.py --config $CFG/ap1_VT_mosei_only_s3407.yaml --mode pretrain"
  "aseq_ap1_avt_cr:python3 scripts/train.py --config $CFG/ap1_AVT_ES_crema_only_s3407.yaml --mode pretrain"
  "aseq_ap1_avt_me:python3 scripts/train.py --config $CFG/ap1_AVT_ES_meld_only_s3407.yaml --mode pretrain"
  "aseq_ap1_avt_mo:python3 scripts/train.py --config $CFG/ap1_AVT_ES_mosei_only_s3407.yaml --mode pretrain"
)

declare -a RUNS_AP2=(
  "aseq_ap2_base:python3 scripts/train.py --config $CFG/ap2_ES_baseline_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_m1:python3 scripts/train.py --config $CFG/ap2_M1_effbatch8_ES_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_m2:python3 scripts/train.py --config $CFG/ap2_M2_lr5e5_ES_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_m3:python3 scripts/train.py --config $CFG/ap2_M3_uniform_ES_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_m4a:python3 scripts/train.py --config $CFG/ap2_M4_plain_ce_ES_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_m4b:python3 scripts/train.py --config $CFG/ap2_M4_focal_ES_3ds_s3407.yaml --mode pretrain"
)

declare -a RUNS_AP2OPT=(
  "aseq_ap2_fcb:python3 scripts/train.py --config $CFG/ap2_opt_fixed_cb_ES_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_lsm:python3 scripts/train.py --config $CFG/ap2_opt_label_smooth_ES_3ds_s3407.yaml --mode pretrain"
  "aseq_ap2_bblr:python3 scripts/train.py --config $CFG/ap2_opt_backbone_lr_ES_3ds_s3407.yaml --mode pretrain"
)

declare -a RUNS_AP3=(
  "aseq_ap3_std:python3 scripts/train.py --config $CFG/ap3_fusion_standard_3ds_s3407.yaml --mode pretrain"
  "aseq_ap3_lf_tx:python3 scripts/train.py --config $CFG/ap3_fusion_leader_text_3ds_s3407.yaml --mode pretrain"
  "aseq_ap3_lf_au:python3 scripts/train.py --config $CFG/ap3_fusion_leader_audio_3ds_s3407.yaml --mode pretrain"
  "aseq_ap3_ts:python3 scripts/train.py --config $CFG/ap3_fusion_two_stage_3ds_s3407.yaml --mode pretrain"
)

declare -a RUNS_AP4=(
  "aseq_ap4_da_def:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_accuracy_seq.yaml --mode pretrain"
  "aseq_ap4_da_w02:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w002_accuracy_seq.yaml --mode pretrain"
  "aseq_ap4_da_w05:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w005_accuracy_seq.yaml --mode pretrain"
  "aseq_ap4_da_w10:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w010_accuracy_seq.yaml --mode pretrain"
  "aseq_ap4_da_w05lr:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_w005_lr5e5_accuracy_seq.yaml --mode pretrain"
  "aseq_ap4_da_uni:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_uniform_accuracy_seq.yaml --mode pretrain"
  "aseq_ap4_da_s34:python3 scripts/train.py --config $CFG/ap4_config_AVT_DA_seed3407_accuracy_seq.yaml --mode pretrain"
)

pick_runs() {
  case "$PHASE" in
    ap0) printf '%s\n' "${RUNS_AP0[@]}" ;;
    ap1) printf '%s\n' "${RUNS_AP1[@]}" ;;
    ap2) printf '%s\n' "${RUNS_AP2[@]}" ;;
    ap2opt) printf '%s\n' "${RUNS_AP2OPT[@]}" ;;
    ap3) printf '%s\n' "${RUNS_AP3[@]}" ;;
    ap4) printf '%s\n' "${RUNS_AP4[@]}" ;;
    all)
      printf '%s\n' "${RUNS_AP0[@]}" "${RUNS_AP1[@]}" "${RUNS_AP2[@]}" "${RUNS_AP2OPT[@]}" "${RUNS_AP3[@]}" "${RUNS_AP4[@]}"
      ;;
    list) printf '%s\n' "${RUNS_AP0[@]}" "${RUNS_AP1[@]}" "${RUNS_AP2[@]}" "${RUNS_AP2OPT[@]}" "${RUNS_AP3[@]}" "${RUNS_AP4[@]}" ;;
    *)
      echo "Usage: $0 {ap0|ap1|ap2|ap2opt|ap3|ap4|all|list}"
      exit 1
      ;;
  esac
}

if [[ "$PHASE" == "list" ]]; then
  echo "# export CUDA_VISIBLE_DEVICES=$GPU_ID"
  pick_runs | while IFS= read -r line; do
    echo "${line#*:}"
  done
  exit 0
fi

if [[ "$PHASE" == "all" ]]; then
  echo "[WARN] phase=all 将启动 $(pick_runs | wc -l) 个 tmux 会话；AVT 多任务同卡极易 OOM，建议分阶段执行。"
fi

while IFS= read -r item; do
  [[ -z "$item" ]] && continue
  session="${item%%:*}"
  cmd="${item#*:}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[INFO] session exists, skip: $session"
    continue
  fi

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
  tmux send-keys -t "$session" "$cmd" C-m
  echo "[OK] $session  (CUDA_VISIBLE_DEVICES=$GPU_ID)"
done < <(pick_runs)

echo
echo "TensorBoard: ./scripts/tensorboard_accuracy_seq.sh"
echo "会话列表: tmux ls"

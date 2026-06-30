#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 全自动重跑总控（主流程分批 + 失败循环补跑，直到清零或达到上限）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
THROTTLED_SCRIPT="$PROJECT_DIR/scripts/start_rerun_all_throttled_gpuaware.sh"
CLEAR_SCRIPT="$PROJECT_DIR/scripts/start_rerun_failed_until_clear.sh"
STATUS_DIR="$PROJECT_DIR/logs_rerun/.launcher_status"
SUMMARIZE_SCRIPT="$PROJECT_DIR/scripts/summarize_rerun_results.py"
TABLE_SCRIPT="$PROJECT_DIR/scripts/build_paper_table_main.py"
RECOMPUTE_TABLE_SCRIPT="$PROJECT_DIR/scripts/recompute_and_fill_paper_table.py"
REPORT_DIR="$PROJECT_DIR/outputs_rerun/autopilot_reports"

GROUP="${1:-all}"          # mainline | expansion | extra | all
MODE="${2:-auto}"          # auto | single | dual
DRY_RUN="${3:-0}"          # 0 | 1
BATCH_SIZE="${4:-2}"       # 节流批次大小
POLL_SEC="${5:-30}"        # 批次轮询秒数
MAX_RETRY="${6:-1}"        # 主流程内单任务重试次数
RETRY_WAIT_SEC="${7:-20}"  # 主流程单任务重试等待秒数
MAX_CLEAR_ROUNDS="${8:-5}" # 失败循环补跑最大轮次
ENABLE_POSTPROC="${9:-1}"  # 1 | 0 训练后自动汇总+出表
ENABLE_RECOMPUTE="${10:-0}" # 1 | 0 是否执行重算回填（较耗时）
RECOMPUTE_BATCH_SIZE="${11:-2}" # 重算脚本 batch_size
AUTO_RERUN_FAILED="${AUTO_RERUN_FAILED:-0}" # 1 | 0 是否自动进入失败循环补跑

if [[ ! -x "$THROTTLED_SCRIPT" ]]; then
  echo "[ERROR] missing executable script: $THROTTLED_SCRIPT"
  exit 1
fi
if [[ ! -x "$CLEAR_SCRIPT" ]]; then
  echo "[ERROR] missing executable script: $CLEAR_SCRIPT"
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[ERROR] DRY_RUN must be 0 or 1"
  exit 1
fi
if ! [[ "$MAX_CLEAR_ROUNDS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] MAX_CLEAR_ROUNDS must be non-negative integer"
  exit 1
fi
if [[ "$AUTO_RERUN_FAILED" != "0" && "$AUTO_RERUN_FAILED" != "1" ]]; then
  echo "[ERROR] AUTO_RERUN_FAILED must be 0 or 1"
  exit 1
fi
if [[ "$ENABLE_POSTPROC" != "0" && "$ENABLE_POSTPROC" != "1" ]]; then
  echo "[ERROR] ENABLE_POSTPROC must be 0 or 1"
  exit 1
fi
if [[ "$ENABLE_RECOMPUTE" != "0" && "$ENABLE_RECOMPUTE" != "1" ]]; then
  echo "[ERROR] ENABLE_RECOMPUTE must be 0 or 1"
  exit 1
fi
if ! [[ "$RECOMPUTE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] RECOMPUTE_BATCH_SIZE must be positive integer"
  exit 1
fi

run_python_step() {
  local py_script="$1"
  shift
  if [[ "${USE_CONDA:-1}" != "0" ]]; then
    local conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    if [[ ! -f "$conda_sh" ]]; then
      if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
          conda_sh="$base/etc/profile.d/conda.sh"
        fi
      fi
    fi
    if [[ ! -f "$conda_sh" ]]; then
      if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="$HOME/anaconda3/etc/profile.d/conda.sh"
      elif [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="/opt/miniconda3/etc/profile.d/conda.sh"
      elif [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="/opt/anaconda3/etc/profile.d/conda.sh"
      fi
    fi
    if [[ ! -f "$conda_sh" ]]; then
      echo "[ERROR] conda script not found for post process."
      echo "        You can set env manually: export CONDA_SH=/path/to/conda.sh"
      return 1
    fi
    # shellcheck disable=SC1090
    source "$conda_sh"
    conda activate myenv310
    python3 "$py_script" "$@"
  else
    python3 "$py_script" "$@"
  fi
}

build_expected_sessions() {
  local group="$1"
  local -n out_ref="$2"
  if [[ "$group" == "mainline" || "$group" == "all" ]]; then
    out_ref+=(
      "rerun_at_noda"
      "rerun_at_da"
      "rerun_vt_noda"
      "rerun_avt_noda"
      "rerun_avt_da"
      "rerun_avt_es"
    )
  fi
  if [[ "$group" == "expansion" || "$group" == "all" ]]; then
    out_ref+=(
      "rerun_t_only"
      "rerun_a_only"
      "rerun_v_only"
      "rerun_avt_da_w002"
      "rerun_avt_da_w005"
      "rerun_avt_da_w010"
    )
  fi
  if [[ "$group" == "extra" || "$group" == "all" ]]; then
    out_ref+=(
      "rerun_avt_noda_uniform"
      "rerun_avt_da_uniform"
      "rerun_avt_noda_lr5e5"
      "rerun_avt_da_w005_lr5e5"
      "rerun_avt_noda_seed3407"
      "rerun_avt_da_seed3407"
    )
  fi
}

echo "[INFO] full autopilot start"
echo "[INFO] group=$GROUP mode=$MODE dry_run=$DRY_RUN batch_size=$BATCH_SIZE poll_sec=$POLL_SEC"
echo "[INFO] max_retry=$MAX_RETRY retry_wait=$RETRY_WAIT_SEC max_clear_rounds=$MAX_CLEAR_ROUNDS"
echo "[INFO] enable_postproc=$ENABLE_POSTPROC enable_recompute=$ENABLE_RECOMPUTE recompute_batch_size=$RECOMPUTE_BATCH_SIZE"
echo "[INFO] auto_rerun_failed=$AUTO_RERUN_FAILED"
echo "[INFO] use_conda=${USE_CONDA:-1} conda_sh=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

echo
echo "=============================="
echo "[STEP 1/2] 主流程分批重跑"
echo "=============================="
bash "$THROTTLED_SCRIPT" "$GROUP" "$MODE" "$DRY_RUN" "$BATCH_SIZE" "$POLL_SEC" "$MAX_RETRY" "$RETRY_WAIT_SEC"

FAILED_LIST="$STATUS_DIR/failed_items_latest.txt"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] dry-run mode: skip step 2 and step 3."
else
  if [[ ! -f "$FAILED_LIST" ]]; then
    echo "[WARN] failed list not found after step 1: $FAILED_LIST"
    echo "[WARN] stop autopilot, please inspect step 1 logs."
    exit 1
  fi

  if [[ "$AUTO_RERUN_FAILED" == "1" && "$MAX_CLEAR_ROUNDS" -gt 0 ]]; then
    echo
    echo "=============================="
    echo "[STEP 2/3] 失败循环补跑直到清零"
    echo "=============================="
    bash "$CLEAR_SCRIPT" "$FAILED_LIST" "$MAX_CLEAR_ROUNDS" "$POLL_SEC" 0
  else
    echo
    echo "=============================="
    echo "[STEP 2/3] 跳过自动失败补跑（由你手动决定）"
    echo "=============================="
    echo "[INFO] failed list kept at: $FAILED_LIST"
  fi
fi

if [[ "$DRY_RUN" != "1" && "$ENABLE_POSTPROC" == "1" ]]; then
  if [[ ! -f "$SUMMARIZE_SCRIPT" || ! -f "$TABLE_SCRIPT" || ! -f "$RECOMPUTE_TABLE_SCRIPT" ]]; then
    echo "[WARN] post-process scripts missing, skip step 3."
    echo "[WARN] expected: summarize_rerun_results.py / build_paper_table_main.py / recompute_and_fill_paper_table.py"
  else
    echo
    echo "=============================="
    echo "[STEP 3/3] 自动汇总与论文表生成"
    echo "=============================="

    run_python_step "$SUMMARIZE_SCRIPT"
    run_python_step "$TABLE_SCRIPT"
    if [[ "$ENABLE_RECOMPUTE" == "1" ]]; then
      run_python_step "$RECOMPUTE_TABLE_SCRIPT" --batch_size "$RECOMPUTE_BATCH_SIZE"
    else
      echo "[INFO] skip recompute table step (ENABLE_RECOMPUTE=0)"
    fi

    echo "[INFO] generated files:"
    echo "       outputs_rerun/rerun_results_summary.csv"
    echo "       outputs_rerun/rerun_results_summary.md"
    echo "       outputs_rerun/paper_table_main.csv"
    echo "       outputs_rerun/paper_table_main.md"
    if [[ "$ENABLE_RECOMPUTE" == "1" ]]; then
      echo "       outputs_rerun/paper_table_main_recomputed.csv"
      echo "       outputs_rerun/paper_table_main_recomputed.md"
    fi
  fi
fi

mkdir -p "$REPORT_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="$REPORT_DIR/autopilot_report_${RUN_ID}.txt"
LATEST_REPORT_FILE="$REPORT_DIR/autopilot_report_latest.txt"

declare -a EXPECTED_SESSIONS=()
build_expected_sessions "$GROUP" EXPECTED_SESSIONS

FAILED_COUNT=0
if [[ -f "$FAILED_LIST" ]]; then
  FAILED_COUNT="$(wc -l < "$FAILED_LIST" | tr -d ' ')"
fi

{
  echo "Autopilot Report"
  echo "run_id=$RUN_ID"
  echo "group=$GROUP mode=$MODE dry_run=$DRY_RUN"
  echo "batch_size=$BATCH_SIZE poll_sec=$POLL_SEC"
  echo "max_retry=$MAX_RETRY retry_wait_sec=$RETRY_WAIT_SEC"
  echo "max_clear_rounds=$MAX_CLEAR_ROUNDS"
  echo "enable_postproc=$ENABLE_POSTPROC enable_recompute=$ENABLE_RECOMPUTE recompute_batch_size=$RECOMPUTE_BATCH_SIZE"
  echo "use_conda=${USE_CONDA:-1} conda_sh=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  echo ""
  echo "[Artifacts]"
  echo "failed_list_latest=$FAILED_LIST"
  echo "summary_csv=$PROJECT_DIR/outputs_rerun/rerun_results_summary.csv"
  echo "summary_md=$PROJECT_DIR/outputs_rerun/rerun_results_summary.md"
  echo "table_csv=$PROJECT_DIR/outputs_rerun/paper_table_main.csv"
  echo "table_md=$PROJECT_DIR/outputs_rerun/paper_table_main.md"
  echo "table_recomputed_csv=$PROJECT_DIR/outputs_rerun/paper_table_main_recomputed.csv"
  echo "table_recomputed_md=$PROJECT_DIR/outputs_rerun/paper_table_main_recomputed.md"
  echo ""
  echo "[Expected Sessions]"
  for s in "${EXPECTED_SESSIONS[@]}"; do
    echo "$s"
  done
  echo ""
  echo "[Pending Sessions From Failed List]"
  if [[ -f "$FAILED_LIST" && "$FAILED_COUNT" -gt 0 ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -z "$line" ]] && continue
      echo "${line%%:*}"
    done < "$FAILED_LIST"
  else
    echo "(none)"
  fi
  echo ""
  echo "[Status]"
  echo "failed_count=$FAILED_COUNT"
  if [[ "$FAILED_COUNT" == "0" ]]; then
    echo "result=all_clear"
  else
    echo "result=has_pending_failures"
  fi
} > "$REPORT_FILE"

cp "$REPORT_FILE" "$LATEST_REPORT_FILE"

echo
echo "[INFO] autopilot report: $REPORT_FILE"
echo "[INFO] latest report  : $LATEST_REPORT_FILE"
echo "[INFO] pending count  : $FAILED_COUNT"
if [[ "$FAILED_COUNT" != "0" ]]; then
  echo "[WARN] pending list   : $FAILED_LIST"
fi
echo "[DONE] full autopilot completed."

#!/usr/bin/env bash
# R4 训练监控 — 从仓库任意目录可用
#
# 用法:
#   bash project/scripts/tail_r4_training.sh status          # 队列 + 当前指标一行摘要
#   bash project/scripts/tail_r4_training.sh meld            # tail MELD F_M_ES 重训日志
#   bash project/scripts/tail_r4_training.sh crema           # tail CREMA F_C_ES 重训日志
#   bash project/scripts/tail_r4_training.sh watch           # tail 综合监视日志
#   bash project/scripts/tail_r4_training.sh metrics meld    # 实时跟踪 metrics.csv
#   bash project/scripts/tail_r4_training.sh tmux meld       # 附着 retrain_meld_es 会话

set -euo pipefail

_resolve_project_dir() {
  local here script_dir repo_root
  here="$(pwd)"
  if [[ -f "$here/project/scripts/tail_r4_training.sh" ]]; then
    echo "$here/project"
    return 0
  fi
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$script_dir/train.py" ]] || [[ -d "$script_dir/config/sdavt_v3_r4" ]]; then
    echo "$script_dir"
    return 0
  fi
  if [[ -f "$script_dir/../project/scripts/train.py" ]]; then
    echo "$(cd "$script_dir/.." && pwd)/project"
    return 0
  fi
  echo "[ERROR] cannot find project/ directory (cwd=$here)" >&2
  exit 1
}

PROJECT_DIR="$(_resolve_project_dir)"
STATUS_DIR="$PROJECT_DIR/outputs_sdavt_v3_r4/status"
LOG_ROOT="$PROJECT_DIR/logs_sdavt_v3_r4"

CMD="${1:-status}"
TARGET="${2:-}"

_print_paths() {
  echo "project_dir: $PROJECT_DIR"
  echo "status_dir:  $STATUS_DIR"
}

_status() {
  cd "$PROJECT_DIR"
  python3 scripts/monitor_sdavt_r4.py --once 2>/dev/null || true
  echo ""
  echo "--- retrain log files ---"
  for f in retrain_meld_es.log retrain_crema_es.log p2_es_retrain_watch.log; do
    local p="$STATUS_DIR/$f"
    if [[ -f "$p" ]]; then
      echo "  [OK] $p ($(wc -l < "$p") lines, mtime $(stat -c %y "$p" 2>/dev/null | cut -d. -f1))"
    else
      echo "  [MISSING] $p"
    fi
  done
  echo ""
  echo "--- active training (metrics.csv mtime) ---"
  for slot in SDAVT_R4_F_M_ES SDAVT_R4_F_C_ES; do
    local m="$LOG_ROOT/$slot/metrics.csv"
    if [[ -f "$m" ]]; then
      local last
      last="$(tail -1 "$m" 2>/dev/null || true)"
      echo "  $slot: $last"
    else
      echo "  $slot: (no metrics yet)"
    fi
  done
  echo ""
  echo "Commands:"
  echo "  bash project/scripts/tail_r4_training.sh meld"
  echo "  bash project/scripts/tail_r4_training.sh metrics meld"
  echo "  tmux attach -t retrain_meld_es"
}

_tail_log() {
  local name="$1"
  local path="$STATUS_DIR/$name"
  mkdir -p "$STATUS_DIR"
  if [[ ! -f "$path" ]]; then
    echo "[WARN] log not found yet: $path"
    echo "       waiting for training to start (or run retrain script with tee)..."
    touch "$path"
  fi
  echo "=== tail -f $path ==="
  tail -f "$path"
}

_metrics_watch() {
  local slot="${1:-SDAVT_R4_F_M_ES}"
  local m="$LOG_ROOT/$slot/metrics.csv"
  echo "Watching $m (Ctrl+C to stop)"
  while true; do
    if [[ -f "$m" ]]; then
      clear 2>/dev/null || true
      echo "=== $slot metrics (last 8 val rows) ==="
      awk -F, '$2=="val"{print}' "$m" | tail -8
      echo ""
      echo "updated: $(stat -c %y "$m" 2>/dev/null | cut -d. -f1)"
    else
      echo "waiting for $m ..."
    fi
    sleep 30
  done
}

_tmux_attach() {
  local sess="${1:-retrain_meld_es}"
  if tmux has-session -t "$sess" 2>/dev/null; then
    exec tmux attach -t "$sess"
  fi
  echo "[ERROR] tmux session '$sess' not found" >&2
  echo "Active sessions:" >&2
  tmux list-sessions 2>/dev/null | grep -E 'retrain|sdavt_r4' || true
  exit 1
}

case "$CMD" in
  status)
    _print_paths
    _status
    ;;
  meld|meld_es|f_m_es)
    _tail_log "retrain_meld_es.log"
    ;;
  crema|crema_es|f_c_es)
    _tail_log "retrain_crema_es.log"
    ;;
  watch|all)
    _tail_log "p2_es_retrain_watch.log"
    ;;
  metrics)
    case "$TARGET" in
      crema|f_c_es) _metrics_watch "SDAVT_R4_F_C_ES" ;;
      *) _metrics_watch "SDAVT_R4_F_M_ES" ;;
    esac
    ;;
  tmux)
    case "$TARGET" in
      crema|f_c_es) _tmux_attach "retrain_crema_es" ;;
      *) _tmux_attach "retrain_meld_es" ;;
    esac
    ;;
  *)
    echo "Usage: $0 {status|meld|crema|watch|metrics [meld|crema]|tmux [meld|crema]}"
    exit 1
    ;;
esac

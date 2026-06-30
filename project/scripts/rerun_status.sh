#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 一键查看重跑状态（controller、活跃会话、完成/失败统计）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
STATUS_DIR="$PROJECT_DIR/logs_rerun/.launcher_status"
CONTROLLER_SESSION="${CONTROLLER_SESSION:-rerun_autopilot}"

echo "=== Rerun Status ==="
echo "project: $PROJECT_DIR"
echo "time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo

echo "[1] Controller Session"
if tmux has-session -t "$CONTROLLER_SESSION" 2>/dev/null; then
  echo "- status : running"
  echo "- name   : $CONTROLLER_SESSION"
  echo "- attach : tmux attach -t $CONTROLLER_SESSION"
else
  echo "- status : stopped"
  echo "- name   : $CONTROLLER_SESSION"
fi
echo

echo "[2] Active tmux sessions (rerun_*)"
active_list="$(tmux ls 2>/dev/null | awk -F: '/^rerun_/ {print $1}' || true)"
if [[ -n "$active_list" ]]; then
  echo "$active_list" | sed 's/^/- /'
else
  echo "- (none)"
fi
echo

echo "[3] Exitcode Summary"
if [[ ! -d "$STATUS_DIR" ]]; then
  echo "- status dir missing: $STATUS_DIR"
  exit 0
fi

total=0
ok=0
fail=0
missing=0

for f in "$STATUS_DIR"/rerun_*.attempt0.exitcode; do
  [[ -e "$f" ]] || continue
  total=$((total + 1))
  code="$(<"$f")"
  if [[ "$code" == "0" ]]; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
done

if [[ "$total" -eq 0 ]]; then
  echo "- no exitcode files yet"
else
  echo "- total : $total"
  echo "- ok    : $ok"
  echo "- fail  : $fail"
fi

if [[ -f "$STATUS_DIR/failed_items_latest.txt" ]]; then
  remain="$(wc -l < "$STATUS_DIR/failed_items_latest.txt" | tr -d ' ')"
  echo "- pending_from_latest: $remain"
else
  echo "- pending_from_latest: (no failed_items_latest.txt)"
fi
echo

echo "[4] Quick Actions"
echo "- 查看总控: tmux attach -t $CONTROLLER_SESSION"
echo "- 查看某训练: tmux attach -t rerun_at_noda"
echo "- 手动补跑失败: ./scripts/start_rerun_failed_from_list.sh logs_rerun/.launcher_status/failed_items_latest.txt 0 20"

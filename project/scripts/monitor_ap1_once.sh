#!/usr/bin/env bash
set -euo pipefail

echo "================ AP1 MONITOR $(date '+%F %T') ================"

echo
echo "[1] tmux 会话状态（AP1）"
tmux ls 2>/dev/null | grep "aseq_ap1" || echo "未发现 aseq_ap1 会话"

echo
echo "[2] GPU 快照（nvidia-smi）"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits || true

echo
echo "[3] AP1 会话最近日志（末尾）"
sessions=$(tmux ls 2>/dev/null | awk -F: '{print $1}' | grep "^aseq_ap1" || true)
if [[ -z "${sessions}" ]]; then
  echo "无 AP1 会话"
else
  for s in ${sessions}; do
    echo "----- ${s} -----"
    tmux capture-pane -pt "${s}" -S -40 | tail -n 25 || true
  done
fi

echo
echo "[4] 错误关键词扫描（AP1 pane 最近日志）"
if [[ -n "${sessions:-}" ]]; then
  for s in ${sessions}; do
    hit=$(tmux capture-pane -pt "${s}" -S -200 | grep -Ei "traceback|cuda out of memory|oom|nan|inf|kept 0 /|error" || true)
    if [[ -n "$hit" ]]; then
      echo "[ALERT] ${s}"
      echo "$hit" | tail -n 8
    fi
  done
else
  echo "无 AP1 会话可扫描"
fi

echo
echo "[5] 最新产物目录（logs/checkpoints）"
echo "logs_accuracy_seq 最近10个："
ls -1dt logs_accuracy_seq/* 2>/dev/null | head -n 10 || echo "暂无日志目录"
echo
echo "checkpoints_accuracy_seq 最近10个："
ls -1dt checkpoints_accuracy_seq/* 2>/dev/null | head -n 10 || echo "暂无checkpoint目录"

echo
echo "================ END AP1 MONITOR ================"

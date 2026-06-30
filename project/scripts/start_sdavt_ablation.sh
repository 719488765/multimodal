#!/usr/bin/env bash
# SDAVT v3 消融实验 — 双 GPU tmux 队列调度
#
# 一键启动:
#   bash scripts/start_sdavt_ablation.sh smoke         # 冒烟测试（须 PASS 后再训练）
#   bash scripts/start_sdavt_ablation.sh fix_v3       # 架构修复后重跑 6 项失败融合
#   bash scripts/start_sdavt_ablation.sh m3           # MELD M3 拉升（fix 通过后）
#   bash scripts/start_sdavt_ablation.sh modal        # MELD 模态消融
#   bash scripts/start_sdavt_ablation.sh fusion       # 完整融合消融 14 runs
#   bash scripts/start_sdavt_ablation.sh all          # fusion + m3 + modal
#   bash scripts/start_sdavt_ablation.sh gen          # 仅生成 yaml + queue.json
#   bash scripts/start_sdavt_ablation.sh status       # 查看队列状态
#   bash scripts/start_sdavt_ablation.sh stop         # 停止 worker

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-myenv310}"
QUEUE_FILE="${QUEUE_FILE:-$PROJECT_DIR/outputs_sdavt_v3/ablation_queue.json}"
PHASE="${1:-all}"

CONDA_SH=""
for _c in \
  "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/usr/local/miniconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -f "$_c" ]]; then CONDA_SH="$_c"; break; fi
done
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _b="$(conda info --base 2>/dev/null || true)"
  [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
fi

_gen_configs() {
  local phase="$1"
  cd "$PROJECT_DIR"
  python3 scripts/generate_sdavt_ablation_configs.py --phase "$phase" --write-queue
}

_status() {
  python3 - <<PY
import json
from pathlib import Path
p = Path("$QUEUE_FILE")
if not p.exists():
    print("Queue not found:", p)
    raise SystemExit(1)
q = json.loads(p.read_text())
from collections import Counter
c = Counter(j.get("status","?") for j in q["jobs"])
print("Queue:", p)
for k,v in sorted(c.items()):
    print(f"  {k}: {v}")
print("--- pending ---")
for j in q["jobs"]:
    if j.get("status") == "pending":
        print(f"  {j['id']} [{j.get('phase')}] gpu_hint={j.get('gpu_hint')}")
PY
}

_stop_workers() {
  for s in sdavt_worker_gpu0 sdavt_worker_gpu1 sdavt_queue; do
    tmux kill-session -t "$s" 2>/dev/null || true
  done
  echo "[OK] workers stopped"
}

_start_tb() {
  tmux kill-session -t sdavt_tb 2>/dev/null || true
  tmux new-session -d -s sdavt_tb "bash $PROJECT_DIR/scripts/tensorboard_sdavt_v3.sh 6007"
}

_run_smoke() {
  [[ -f "$CONDA_SH" ]] || { echo "[ERROR] conda.sh not found"; exit 1; }
  cd "$PROJECT_DIR"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
  echo "[smoke] MOSEI S4 + MELD LFT + MELD S2-M1 ..."
  python3 scripts/validate_sdavt_v3_training_smoke.py --all --steps 50
  python3 scripts/validate_two_stage_fusion.py
  echo "[OK] smoke tests passed"
}

_launch_workers() {
  [[ -f "$CONDA_SH" ]] || { echo "[ERROR] conda.sh not found"; exit 1; }
  [[ -f "$QUEUE_FILE" ]] || { echo "[ERROR] queue missing; run: $0 gen"; exit 1; }

  _stop_workers
  _start_tb

  for GPU in 0 1; do
    local sess="sdavt_worker_gpu${GPU}"
    tmux new-session -d -s "$sess"
    tmux send-keys -t "$sess" "cd \"$PROJECT_DIR\"" C-m
    tmux send-keys -t "$sess" "source \"$CONDA_SH\"" C-m
    tmux send-keys -t "$sess" "conda activate $ENV_NAME" C-m
    tmux send-keys -t "$sess" "export CUDA_VISIBLE_DEVICES=$GPU" C-m
    tmux send-keys -t "$sess" "bash scripts/sdavt_ablation_worker.sh $GPU \"$QUEUE_FILE\"" C-m
    echo "[OK] worker $sess GPU=$GPU"
  done
}

case "$PHASE" in
  smoke)
    _run_smoke
    ;;
  gen)
    _gen_configs all
    ;;
  fix|fix_v3|fusion|m3|modal)
    _gen_configs "$PHASE"
    _launch_workers
    ;;
  all)
    _gen_configs all
    _launch_workers
    ;;
  status)
    _status
    tmux list-sessions 2>/dev/null | grep -E 'sdavt_' || true
    ;;
  stop)
    _stop_workers
    ;;
  *)
    echo "Usage: $0 {smoke|fix|fix_v3|m3|modal|fusion|all|gen|status|stop}"
    exit 1
    ;;
esac

echo ""
echo "Monitor: tmux attach -t sdavt_worker_gpu0 | sdavt_worker_gpu1"
echo "TensorBoard: http://<host>:6007"
echo "Status: bash scripts/start_sdavt_ablation.sh status"

#!/usr/bin/env bash
# 融合消融 14 run 全部完成后：汇总选型 → 启动 MELD M3 主流训练
#
# 用法:
#   bash scripts/start_sdavt_after_fusion.sh check   # 仅检查融合是否完成
#   bash scripts/start_sdavt_after_fusion.sh run     # 完成后自动启动 M3（6 runs）

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
QUEUE_FILE="$PROJECT_DIR/outputs_sdavt_v3/ablation_queue.json"
PHASE="${1:-check}"

cd "$PROJECT_DIR"

_count_pending_fusion() {
  python3 - <<PY
import json
from pathlib import Path
p = Path("$QUEUE_FILE")
if not p.exists():
    print("missing"); raise SystemExit(2)
q = json.loads(p.read_text())
fusion = [j for j in q["jobs"] if j.get("phase") == "fusion"]
pending = sum(1 for j in fusion if j.get("status") == "pending")
running = sum(1 for j in fusion if j.get("status") == "running")
done = sum(1 for j in fusion if j.get("status") == "done")
failed = sum(1 for j in fusion if j.get("status") == "failed")
print(f"fusion total={len(fusion)} done={done} running={running} pending={pending} failed={failed}")
if pending + running > 0:
    raise SystemExit(1)
PY
}

_pick_meld_fusion() {
  python3 - <<'PY'
import csv
import re
from pathlib import Path

log = Path("logs_sdavt_v3")
best_name, best_f1, best_fusion = None, -1.0, "emotion_shift"
for m in log.glob("SDAVT_F_M_*/metrics.csv"):
    rows = [r for r in csv.DictReader(m.open()) if r.get("phase") == "val" and r.get("f1")]
    if not rows:
        continue
    f1 = max(float(r["f1"]) for r in rows)
    if f1 > best_f1:
        best_f1, best_name = f1, m.parent.name
        name = m.parent.name.upper()
        if re.search(r"_STD_|STANDARD", name):
            best_fusion = "standard"
        elif re.search(r"_LFT_|_LFA_|LEADER", name):
            best_fusion = "leader_follower"
        elif re.search(r"_TS_|TWO", name):
            best_fusion = "two_stage"
        else:
            best_fusion = "emotion_shift"
print(best_fusion)
import sys
if best_name:
    print(f"# best: {best_name} f1={best_f1:.4f}", file=sys.stderr)
PY
}

case "$PHASE" in
  check)
    if _count_pending_fusion; then
      echo "[OK] 融合 14 run 已全部结束，可执行: bash scripts/start_sdavt_after_fusion.sh run"
      python3 scripts/build_sdavt_ablation_table.py
      echo "--- MELD 推荐 fusion（按 best F1）---"
      _pick_meld_fusion || true
    else
      echo "[WAIT] 融合实验尚未完成，请稍后重试"
      bash scripts/start_sdavt_ablation.sh status
      exit 1
    fi
    ;;
  run)
    _count_pending_fusion || { echo "[ABORT] 融合未完成"; exit 1; }
    echo "[1/3] 汇总融合结果"
    python3 scripts/summarize_sdavt_v3_results.py
    python3 scripts/build_sdavt_ablation_table.py

    MELD_FUSION="$(_pick_meld_fusion | head -1)"
    echo "[2/3] MELD 最优融合: $MELD_FUSION → 启动 M3"
    echo "[3/3] 启动 M3 worker（双卡）"
    bash scripts/start_sdavt_ablation.sh m3
    echo "[OK] M3 训练已启动。完成后: bash scripts/start_sdavt_ablation.sh modal"
    ;;
  *)
    echo "Usage: $0 {check|run}"
    exit 1
    ;;
esac

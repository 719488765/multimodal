#!/usr/bin/env bash
# Monitor M3_M1 / M3_M7 anti-overfit retrain: ep3 peak + ep8 unfreeze stability
# DISABLED BY DEFAULT — opt-in: ENABLE_R4_WATCH=1 bash scripts/watch_p3_antioverfit_milestones.sh
set -euo pipefail

if [[ "${ENABLE_R4_WATCH:-0}" != "1" ]]; then
  echo "[watch_p3_antioverfit] DISABLED (default). Set ENABLE_R4_WATCH=1 to run."
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

LOG="${LOG:-outputs_sdavt_v3_r4/status/p3_antioverfit_milestone_watch.log}"
INTERVAL="${INTERVAL:-300}"
STATUS_DIR="outputs_sdavt_v3_r4/status/milestone_flags"
UNFREEZE_EPOCH=8

mkdir -p "$STATUS_DIR"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

_analyze() {
  python3 - "$@" <<'PY'
import csv, sys
from pathlib import Path

job_id, run_dir, target_ep, milestone = sys.argv[1:5]
f1_tgt, acc_tgt, champ_f1, unfreeze_ep = 0.59, 0.62, 0.6105, 8
target_ep = int(target_ep)

p = Path("logs_sdavt_v3_r4") / run_dir / "metrics.csv"
if not p.exists():
    raise SystemExit(1)

val = {}
with p.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("phase") != "val":
            continue
        val[int(row["epoch"])] = {
            "f1": float(row["f1"]),
            "acc": float(row["accuracy"]),
            "loss": float(row["loss"]),
        }

if target_ep not in val:
    raise SystemExit(1)

cur = val[target_ep]
best_ep = max(val, key=lambda e: val[e]["f1"])
best = val[best_ep]

print(f"=== {milestone} | {job_id} | epoch {target_ep} ===")
print(f"  val_f1={cur['f1']:.4f} val_acc={cur['acc']:.4f} val_loss={cur['loss']:.4f}")
print(f"  best_so_far: ep{best_ep} f1={best['f1']:.4f} acc={best['acc']:.4f}")
print(f"  tier2_f1: {'PASS' if cur['f1'] >= f1_tgt else 'below'} ({cur['f1']:.4f} vs {f1_tgt})")
print(f"  tier2_acc: {'PASS' if cur['acc'] >= acc_tgt else 'below'} ({cur['acc']:.4f} vs {acc_tgt})")
print(f"  vs_champion_M3_M3: f1_delta={cur['f1'] - champ_f1:+.4f}")

if milestone == "EP8_UNFREEZE":
    pre = [val[e]["f1"] for e in sorted(val) if e < unfreeze_ep]
    post = [val[e]["f1"] for e in sorted(val) if e >= unfreeze_ep]
    if pre and post:
        pre_best = max(pre)
        post_last = post[-1]
        collapse = post_last < pre_best - 0.08
        print(f"  pre_unfreeze_best_f1={pre_best:.4f}")
        print(f"  post_unfreeze_last_f1={post_last:.4f}")
        print(f"  collapse_detected: {'YES — investigate' if collapse else 'NO — stable so far'}")
PY
}

_check() {
  local job_id="$1" run_dir="$2"
  local metrics="logs_sdavt_v3_r4/${run_dir}/metrics.csv"
  [[ -f "$metrics" ]] || return 0

  for spec in "3:EP3_CHECK" "8:EP8_UNFREEZE"; do
    local ep="${spec%%:*}" ms="${spec#*:}"
    local flag="${STATUS_DIR}/${job_id}_${ms}.done"
    [[ -f "$flag" ]] && continue
    if out="$(_analyze "$job_id" "$run_dir" "$ep" "$ms" 2>/dev/null)"; then
      while IFS= read -r line; do [[ -n "$line" ]] && log "$line"; done <<< "$out"
      touch "$flag"
    fi
  done

  local done_flag="${STATUS_DIR}/${job_id}_TRAINING_DONE.done"
  [[ -f "$done_flag" ]] && return 0
  local st
  st="$(python3 -c "import json; q=json.load(open('outputs_sdavt_v3_r4/experiment_queue.json')); print(next(j['status'] for j in q['jobs'] if j['id']=='$job_id'))")"
  if [[ "$st" == "done" || "$st" == "failed" ]]; then
    log "=== TRAINING_END | ${job_id} | status=${st} ==="
    python3 scripts/accept_sdavt_r4_p3_tier2.py --refresh-report >> "$LOG" 2>&1 || true
    touch "$done_flag"
  fi
}

log "milestone watcher started interval=${INTERVAL}s unfreeze_epoch=${UNFREEZE_EPOCH}"

while true; do
  _check "M3_M1_roberta" "SDAVT_R4_M3_M1_roberta"
  _check "M3_M7_combo" "SDAVT_R4_M3_M7_combo"

  if [[ -f "${STATUS_DIR}/M3_M1_roberta_TRAINING_DONE.done" && -f "${STATUS_DIR}/M3_M7_combo_TRAINING_DONE.done" ]]; then
    log "All jobs finished — watcher exit"
    exit 0
  fi
  sleep "$INTERVAL"
done

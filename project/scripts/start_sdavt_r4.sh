#!/usr/bin/env bash
# SDAVT R4 orchestration: queue gen, worker launch, status.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

QUEUE="outputs_sdavt_v3_r4/experiment_queue.json"
ENV_NAME="${ENV_NAME:-myenv310}"

_activate_conda() {
  local CONDA_SH=""
  for _c in \
    "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [[ -f "$_c" ]] && CONDA_SH="$_c" && break
  done
  if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
    local _b
    _b="$(conda info --base 2>/dev/null || true)"
    [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
  fi
  [[ -n "$CONDA_SH" ]] || return 1
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
}

_gen_p4_crema_mosei() {
  python3 - <<'PY'
import json
from datetime import datetime
from pathlib import Path

import yaml

root = Path("config/sdavt_v3_r4/p4_modal")
qpath = Path("outputs_sdavt_v3_r4/experiment_queue.json")
q = json.loads(qpath.read_text(encoding="utf-8"))
existing = {j["id"] for j in q["jobs"]}
added = 0
priority = 50
for dataset in ("crema", "mosei"):
    cfg_dir = root / dataset
    if not cfg_dir.is_dir():
        continue
    for cfg in sorted(cfg_dir.glob("R4_A_*.yaml")):
        meta = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        exp = meta.get("experiment") or {}
        job_id = exp.get("job_id") or cfg.stem.replace("_emotion_shift", "")
        if job_id in existing:
            continue
        modality = job_id.split("_")[-1] if "_" in job_id else "AVT"
        q["jobs"].append(
            {
                "id": job_id,
                "phase": "p4_modal",
                "dataset": dataset,
                "modality": modality,
                "fusion": "emotion_shift",
                "config": str(cfg).replace("\\", "/"),
                "priority": priority,
                "gpu_hint": 1 if dataset == "crema" else 0,
                "paper_section": f"Table4-{dataset.upper()} 模态消融",
                "depends_on": "p3_m3_winner_meld",
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "run_dir": None,
                "best_val_f1": None,
                "best_val_acc": None,
                "note": "p4_crema_mosei",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        existing.add(job_id)
        priority += 1
        added += 1
qpath.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[OK] added {added} p4 crema/mosei jobs")
PY
}

_run_workers() {
  local phase="${1:-all}"
  _activate_conda || true
  for s in sdavt_r4_worker_gpu0 sdavt_r4_worker_gpu1; do
    tmux kill-session -t "$s" 2>/dev/null || true
  done
  for GPU in 0 1; do
    tmux new-session -d -s "sdavt_r4_worker_gpu${GPU}"
    tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "cd \"$PROJECT_DIR\"" C-m
    tmux send-keys -t "sdavt_r4_worker_gpu${GPU}" "bash scripts/sdavt_r4_worker.sh $GPU" C-m
  done
  echo "[OK] workers started (phase hint: $phase)"
}

_status() {
  python3 scripts/monitor_sdavt_r4.py --once 2>/dev/null || python3 scripts/build_sdavt_r4_report.py
}

cmd="${1:-status}"
case "$cmd" in
  gen)
    sub="${2:-}"
    case "$sub" in
      p4_crema_mosei) _gen_p4_crema_mosei ;;
      *) echo "Usage: $0 gen p4_crema_mosei"; exit 1 ;;
    esac
    ;;
  run)
    sub="${2:-all}"
    _run_workers "$sub"
    ;;
  status) _status ;;
  report) python3 scripts/build_sdavt_r4_report.py ;;
  *)
    echo "Usage: $0 {gen p4_crema_mosei|run [phase]|status|report}"
    exit 1
    ;;
esac

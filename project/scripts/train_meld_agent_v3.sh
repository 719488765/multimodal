#!/usr/bin/env bash
# MELD v3 RoBERTa 架构升级训练（P3）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-config/rerun/accuracy_plan/ap2_M1_meld_only_agent_v3_roberta.yaml}"
export SKIP_AUDIO_EXTRACT=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

exec "$ROOT/scripts/train_meld_agent_v2.sh"

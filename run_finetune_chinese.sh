#!/usr/bin/env bash
# 从仓库根目录 multimodal/ 一键后台微调（转发到 project/scripts/）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/project/scripts/run_finetune_chinese_background.sh"

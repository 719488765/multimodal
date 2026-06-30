#!/usr/bin/env bash
# 仓库根目录快捷入口 → project/scripts/tail_r4_training.sh
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/project/scripts/tail_r4_training.sh" "$@"

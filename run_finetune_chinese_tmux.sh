#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/project/scripts/run_finetune_chinese_tmux.sh" "$@"

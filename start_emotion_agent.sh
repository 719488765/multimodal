#!/usr/bin/env bash
# 从 multimodal 根目录一键启动 Emotion Agent 全栈
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/emotion-agent/scripts/start_full_stack.sh" "$@"

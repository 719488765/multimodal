#!/usr/bin/env bash
# Start Ollama and ensure the chat model from backend/.env is pulled.
set -euo pipefail

MODEL="${LLM_MODEL:-qwen2.5:7b-instruct}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not found. Install from https://ollama.com" >&2
  exit 1
fi

if ! curl -sf "http://${HOST}/api/tags" >/dev/null 2>&1; then
  echo "Starting ollama serve..."
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    if curl -sf "http://${HOST}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -sf "http://${HOST}/api/tags" | grep -q "\"name\""; then
  echo "Pulling model ${MODEL} (first time may take several minutes)..."
  ollama pull "${MODEL}"
fi

echo "Ollama ready. Models:"
curl -s "http://${HOST}/api/tags" | python3 -m json.tool 2>/dev/null || curl -s "http://${HOST}/api/tags"
echo ""
echo "Backend .env should have: LLM_PROVIDER=ollama LLM_MODEL=${MODEL}"

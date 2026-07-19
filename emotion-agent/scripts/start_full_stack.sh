#!/usr/bin/env bash
# 从零一键启动 Emotion Agent 全栈：
#   环境检查/首次安装 → Ollama(LLM) → ASR(9010) → 前端构建 → 后端(8000) → Cloudflare 隧道
#
# 用法:
#   cd emotion-agent && ./scripts/start_full_stack.sh
#
# 常用环境变量:
#   SKIP_SETUP=1          跳过首次依赖安装
#   SKIP_TUNNEL=1         不启动 cloudflared（仅本机/Cursor 转发）
#   BUILD_MODE=server     构建模式: server(默认, cloudflared/nginx) | cursor(Cursor 端口转发)
#   FORCE_RESTART=1       强制重启已有 tmux 会话与 8000/9010 端口
#   APPLY_CHINESE_PRESET=1  若存在微调 checkpoint，自动 apply_agent_chinese_preset.sh
#   TMUX_SESSION=emotion-full  主 tmux 会话名
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$ROOT/../project" && pwd)}"
SESSION="${TMUX_SESSION:-emotion-full}"
CF_SESSION="${CF_TMUX_SESSION:-cf_tunnel}"
PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
CONDA_PY="$PYTHON"

SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_TUNNEL="${SKIP_TUNNEL:-0}"
BUILD_MODE="${BUILD_MODE:-server}"
FORCE_RESTART="${FORCE_RESTART:-0}"
APPLY_CHINESE_PRESET="${APPLY_CHINESE_PRESET:-0}"
OLLAMA_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct}"
CF_LOG="${HOME}/cloudflared_tunnel.log"

log()  { echo "==> $*"; }
warn() { echo "WARN: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

wait_url() {
  local url="$1" label="$2" max="${3:-90}" i=1
  while [[ $i -le $max ]]; do
    if curl -sf --max-time 5 "$url" >/dev/null 2>&1; then
      echo "    $label 就绪 (${i}s)"
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  return 1
}

extract_cf_url() {
  grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$CF_LOG" 2>/dev/null | tail -1 || true
}

# ---------------------------------------------------------------------------
# 0. 前置检查
# ---------------------------------------------------------------------------
log "Emotion Agent 全栈启动"
echo "    项目目录: $ROOT"
echo "    训练工程: $PROJECT_ROOT"
echo ""

for cmd in tmux curl ss npm node; do
  command -v "$cmd" >/dev/null 2>&1 || die "未找到 $cmd，请先安装"
done

if [[ ! -x "$CONDA_PY" ]]; then
  die "未找到 conda Python: $CONDA_PY（请 conda activate myenv310 或设置 PYTHON=）"
fi

# ---------------------------------------------------------------------------
# 1. 首次环境准备（可 SKIP_SETUP=1 跳过）
# ---------------------------------------------------------------------------
if [[ "$SKIP_SETUP" != "1" ]]; then
  log "1/N 首次环境检查与依赖安装"

  if [[ ! -f "$ROOT/backend/.env" ]]; then
    if [[ -f "$ROOT/deploy/.env.example" ]]; then
      cp "$ROOT/deploy/.env.example" "$ROOT/backend/.env"
      warn "已从 deploy/.env.example 复制 backend/.env，请按需编辑"
    else
      warn "backend/.env 不存在，请手动创建"
    fi
  fi

  if ! "$CONDA_PY" -c "import uvicorn" 2>/dev/null; then
    log "    安装 backend 依赖..."
    "$CONDA_PY" -m pip install -q -r "$ROOT/backend/requirements.txt"
  fi

  if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
    log "    安装 frontend 依赖 (npm install)..."
    (cd "$ROOT/frontend" && npm install)
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "未检测到 ffmpeg，ASR 解码 webm/wav 可能失败。安装: sudo apt-get install -y ffmpeg"
  fi

  log "    检查 ASR 依赖 (faster-whisper)..."
  chmod +x "$ROOT/asr-local/start_server.sh"
  # start_server.sh 会在首次运行时自动 pip install

  log "    检查 Ollama 与模型 ${OLLAMA_MODEL}..."
  chmod +x "$ROOT/scripts/start_ollama.sh"
  export LLM_MODEL="$OLLAMA_MODEL"
  "$ROOT/scripts/start_ollama.sh" || warn "Ollama 未就绪，LLM 将降级为 template"
else
  log "1/N 跳过首次安装 (SKIP_SETUP=1)"
fi

# ---------------------------------------------------------------------------
# 2. 可选：链接中文微调 checkpoint
# ---------------------------------------------------------------------------
if [[ "$APPLY_CHINESE_PRESET" == "1" ]]; then
  PRESET_SCRIPT="$PROJECT_ROOT/scripts/apply_agent_chinese_preset.sh"
  if [[ -x "$PRESET_SCRIPT" ]]; then
    log "2/N 应用 agent_chinese preset"
    (cd "$PROJECT_ROOT" && "$PRESET_SCRIPT") || warn "apply_agent_chinese_preset 失败，继续使用现有 .env"
  else
    warn "未找到 $PRESET_SCRIPT，跳过 preset"
  fi
else
  log "2/N 跳过 preset（设 APPLY_CHINESE_PRESET=1 可自动链接中文微调 checkpoint）"
fi

# ---------------------------------------------------------------------------
# 3. 启动 Ollama
# ---------------------------------------------------------------------------
log "3/N 启动 Ollama (11434)"
chmod +x "$ROOT/scripts/start_ollama.sh" 2>/dev/null || true
export LLM_MODEL="$OLLAMA_MODEL"
"$ROOT/scripts/start_ollama.sh" || warn "Ollama 未就绪"

# ---------------------------------------------------------------------------
# 4. 清理旧会话（FORCE_RESTART）
# ---------------------------------------------------------------------------
if [[ "$FORCE_RESTART" == "1" ]]; then
  log "4/N 强制重启：停止旧 tmux 会话与端口"
  for s in "$SESSION" emotion-demo; do
    tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
  done
  if [[ "$SKIP_TUNNEL" != "1" ]]; then
    tmux has-session -t "$CF_SESSION" 2>/dev/null && tmux kill-session -t "$CF_SESSION" || true
  fi
  # Never use fuser/ps — they hang and fork-storm CPU on this host.
  bash "$ROOT/scripts/safe_free_port.sh" 8000 9010 || true
  sleep 2
elif tmux has-session -t "$SESSION" 2>/dev/null; then
  echo ""
  echo "tmux 会话 '$SESSION' 已存在。"
  echo "  进入:   tmux attach -t $SESSION"
  echo "  重建:   FORCE_RESTART=1 $0"
  CF_URL="$(extract_cf_url)"
  [[ -n "$CF_URL" ]] && echo "  公网:   $CF_URL"
  echo "  本机:   http://127.0.0.1:8000"
  exit 0
else
  log "4/N 无旧会话，继续启动"
fi

# ---------------------------------------------------------------------------
# 5. tmux：ASR + 后端演示
# ---------------------------------------------------------------------------
log "5/N 创建 tmux 会话: $SESSION"

tmux new-session -d -s "$SESSION" -n asr -c "$ROOT/asr-local"
tmux send-keys -t "$SESSION:asr" "./start_server.sh" Enter

tmux new-window -t "$SESSION" -n demo -c "$ROOT"
BUILD_CMD="./scripts/build_production.sh"
if [[ "$BUILD_MODE" == "cursor" ]]; then
  BUILD_CMD="./scripts/build_cursor.sh"
fi
tmux send-keys -t "$SESSION:demo" \
  "echo '等待 ASR 9010...'; for i in \$(seq 1 90); do curl -sf http://127.0.0.1:9010/health >/dev/null && break; sleep 2; done; ${BUILD_CMD}; FORCE_RESTART=1 ./scripts/start_demo.sh" Enter

log "    等待 ASR (9010)..."
if ! wait_url "http://127.0.0.1:9010/health" "ASR" 120; then
  warn "ASR 120s 内未就绪，请 tmux attach -t $SESSION 查看 asr 窗口"
fi

log "    等待后端 (8000，加载 GPU 模型可能 1–3 分钟)..."
if ! wait_url "http://127.0.0.1:8000/api/v1/health" "Backend" 240; then
  warn "Backend 240s 内未就绪，请 tmux attach -t $SESSION 查看 demo 窗口"
fi

# ---------------------------------------------------------------------------
# 6. Cloudflare Quick Tunnel
# ---------------------------------------------------------------------------
CF_URL=""
if [[ "$SKIP_TUNNEL" != "1" ]]; then
  log "6/N 启动 Cloudflare Quick Tunnel"
  chmod +x "$ROOT/scripts/start_cloudflared_tunnel.sh" 2>/dev/null || true
  if "$ROOT/scripts/start_cloudflared_tunnel.sh" "http://127.0.0.1:8000"; then
    for _ in $(seq 1 15); do
      CF_URL="$(extract_cf_url)"
      [[ -n "$CF_URL" ]] && break
      sleep 2
    done
  else
    warn "cloudflared 启动失败，可 SKIP_TUNNEL=1 仅本机访问"
  fi
else
  log "6/N 跳过 cloudflared (SKIP_TUNNEL=1)"
fi

# ---------------------------------------------------------------------------
# 7. 健康检查与冒烟测试
# ---------------------------------------------------------------------------
log "7/N 健康检查"

echo ""
echo "--- ASR ---"
curl -s http://127.0.0.1:9010/health 2>/dev/null | python3 -m json.tool 2>/dev/null || warn "ASR health 失败"

echo ""
echo "--- Backend /api/v1/health ---"
HEALTH_JSON="$(curl -s http://127.0.0.1:8000/api/v1/health 2>/dev/null || echo '{}')"
echo "$HEALTH_JSON" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_JSON"

echo ""
echo "--- 文本推理冒烟 (我很难过) ---"
SMOKE="$(curl -s -X POST http://127.0.0.1:8000/api/v1/emotion/infer-upload \
  -F "session_id=full_stack_smoke" \
  -F "text=我很难过" \
  -F 'metadata={"source":"start_full_stack"}' 2>/dev/null || echo '{}')"
echo "$SMOKE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('  label:', d.get('emotion_label'), '| conf:', round(d.get('confidence', 0), 3))
    probs = d.get('all_probs') or []
    if len(probs) > 1:
        print('  sad prob:', round(probs[1], 3))
    print('  preset:', d.get('checkpoint_preset'), '| source:', d.get('inference_source'))
except Exception as e:
    print('  (冒烟跳过:', e, ')')
" 2>/dev/null || warn "文本推理冒烟失败"

# ---------------------------------------------------------------------------
# 8. 汇总
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Emotion Agent 全栈已启动"
echo "=============================================="
echo ""
echo "  tmux 主会话:  tmux attach -t $SESSION"
echo "    窗口 0 asr  — Whisper ASR :9010"
echo "    窗口 1 demo — 前端+后端   :8000"
echo ""
echo "  本机访问:     http://127.0.0.1:8000"
echo "  Cursor 转发:  Ports 面板转发 8000 → 本机浏览器打开同上"
if [[ -n "$CF_URL" ]]; then
  echo "  公网 HTTPS:   $CF_URL"
  echo "                （Quick Tunnel，重启后 URL 会变）"
fi
if [[ "$SKIP_TUNNEL" == "1" ]]; then
  echo "  公网隧道:     未启动（SKIP_TUNNEL=1）"
  echo "                手动: ./scripts/start_cloudflared_tunnel.sh"
fi
echo ""
echo "  Ollama LLM:   http://127.0.0.1:11434  (model: ${OLLAMA_MODEL})"
echo ""
echo "  浏览器测试:"
echo "    1. 打开上述地址，硬刷新 Ctrl+F5"
echo "    2. 确认顶部 ASR ✓ / LLM ✓ / 模型 checkpoint ✓"
echo "    3. 启用摄像头与麦克风 → 说话 → 结束采集并推理"
echo ""
echo "  常用命令:"
echo "    curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool"
echo "    ./backend/scripts/verify_trained_model.sh"
echo "    ./scripts/stop_full_stack.sh"
echo ""
echo "  停止全部:     ./scripts/stop_full_stack.sh"
echo "  强制重建:     FORCE_RESTART=1 ./scripts/start_full_stack.sh"
echo "=============================================="

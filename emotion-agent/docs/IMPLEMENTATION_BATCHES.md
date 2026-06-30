# Emotion Agent 改进实施批次计划

> 对应方案：视频 clip 对齐 + bert-base-chinese 微调 + LLM 概率注入  
> 状态：✅ Batch 1–4 代码已落地 · Batch 4 微调需 GPU 手动执行

---

## Batch 1：视频推理对齐（后端） ✅

| 任务 | 文件 | 状态 |
|------|------|------|
| 共享抽帧函数 | `project/utils/video_frame_utils.py` | ✅ |
| `preprocess_video_from_file_bytes` | `project/utils/emotion_inference_service.py` | ✅ |
| `predict_from_sample` 按 mime 分流 | 同上 | ✅ |
| routes metadata 传递 video_mime | `backend/app/api/routes.py` | ✅ |

---

## Batch 2：视频采集（前端） ✅

| 任务 | 文件 | 状态 |
|------|------|------|
| MediaRecorder 同步录制 webm | `frontend/src/App.jsx` | ✅ |
| infer 上传 video/webm + metadata | `frontend/src/api.js` | ✅ |
| pipeline_trace 展示 decode_mode | `frontend/src/App.jsx` | ✅ |
| JPEG 单帧 fallback | `frontend/src/App.jsx` | ✅ |

---

## Batch 3：LLM 与模型对齐 ✅

| 任务 | 文件 | 状态 |
|------|------|------|
| AgentRespondRequest 扩展 | `backend/app/models/schemas.py` | ✅ |
| `_build_messages` 注入概率 | `backend/app/services/llm_service.py` | ✅ |
| 前端传完整 emo 字段 | `frontend/src/App.jsx` | ✅ |
| UI 展示回复依据 | `frontend/src/App.jsx` | ✅ |

---

## Batch 4：bert-base-chinese 微调与部署 ✅（脚本就绪，训练待跑）

| 任务 | 文件 | 状态 |
|------|------|------|
| 训练 config | `project/config/rerun/accuracy_plan/ap2_M1_chinese_text_agent.yaml` | ✅ |
| `--skip_text_encoder` | `project/scripts/train.py` | ✅ |
| `load_checkpoint_partial` | `project/utils/helpers.py` | ✅ |
| 微调脚本 | `project/scripts/finetune_agent_chinese.sh` | ✅ |
| preset `agent_chinese` | `backend/app/core/config.py` | ✅ |

**微调命令（需 GPU + 数据）：**

```bash
conda activate myenv310
cd project
chmod +x scripts/finetune_agent_chinese.sh
./scripts/finetune_agent_chinese.sh
# 完成后将 best checkpoint 链到 preset 路径，或 .env 设置：
# MODEL_CHECKPOINT_PRESET=agent_chinese
# MODEL_CHECKPOINT_PATH=/path/to/checkpoint_finetune_best_f1.pth
```

---

## Batch 5：文档与一键启动 ✅

| 任务 | 文件 | 状态 |
|------|------|------|
| 架构文档 | `docs/ARCHITECTURE.md` | ✅ |
| 启动手册 | `START_DEMO.txt` | ✅ |
| **全栈一键启动** | `scripts/start_full_stack.sh` | ✅ |
| **全栈一键停止** | `scripts/stop_full_stack.sh` | ✅ |
| 根目录快捷入口 | `../start_emotion_agent.sh` | ✅ |
| tmux 分步启动（旧） | `scripts/start_all_demo.sh` | ✅ |
| Cloudflare 隧道 | `scripts/start_cloudflared_tunnel.sh` | ✅ |

---

## Batch 6：公网部署与上传优化 ✅

| 任务 | 文件 | 状态 |
|------|------|------|
| nginx + systemd 模板 | `deploy/nginx.conf`, `deploy/systemd/` | ✅ |
| 部署文档 | `deploy/SERVER_DEPLOY.md` | ✅ |
| 生产构建 | `scripts/build_production.sh`, `VITE_DEPLOY_MODE=server` | ✅ |
| Cursor 隧道构建 | `scripts/build_cursor.sh`, `VITE_DEPLOY_MODE=tunnel` | ✅ |
| 整包 multipart 优先 | `frontend/src/api.js` | ✅ |
| 服务器模式保留 webm | `frontend/src/App.jsx` | ✅ |
| infer-upload metadata | `backend/app/api/routes.py` | ✅ |
| BERT 顶层解冻（微调） | `feature_extractors.py` + yaml | ✅ |

---

## 验收清单

- [x] 代码：trace 支持 `decode_mode=video_file`
- [x] 代码：LLM prompt 含 all_probs
- [ ] 运行：浏览器采集产生 webm clip（需重启 8000 + 硬刷新）
- [ ] 运行：微调完成后 `agent_chinese` preset 加载成功
- [x] 代码：公网 server 模式整包上传
- [ ] 运行：公网 HTTPS 下 webm 4 帧推理

---

## 一键启动（推荐）

从零启动 **Ollama · ASR · 后端+前端 · Cloudflare 隧道**，并自动做健康检查与文本推理冒烟。

### 方式 A：在 `emotion-agent/` 目录

```bash
cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent
chmod +x scripts/start_full_stack.sh scripts/stop_full_stack.sh
./scripts/start_full_stack.sh
```

### 方式 B：在 `multimodal/` 根目录

```bash
cd /home/lizhichun_24/sda1/code/multimodal
./start_emotion_agent.sh
```

### 脚本自动完成的步骤

| 顺序 | 组件 | 端口 | 说明 |
|------|------|------|------|
| 1 | 环境检查 | — | 首次自动 `pip install` / `npm install`（可用 `SKIP_SETUP=1` 跳过） |
| 2 | Ollama (LLM) | 11434 | 拉取 `qwen2.5:7b-instruct` |
| 3 | ASR (Whisper) | 9010 | tmux 窗口 `asr` |
| 4 | 后端 + 前端静态页 | 8000 | 生产构建 → uvicorn + `frontend/dist`，tmux 窗口 `demo` |
| 5 | Cloudflare Quick Tunnel | — | tmux 会话 `cf_tunnel`，输出 `https://xxx.trycloudflare.com` |
| 6 | 健康检查 | — | ASR / backend health + 文本「我很难过」冒烟 |

启动完成后终端会打印：

- 本机：`http://127.0.0.1:8000`
- 公网 HTTPS：`https://xxx.trycloudflare.com`（Quick Tunnel，重启后 URL 会变）
- tmux：`tmux attach -t emotion-full`（窗口 0=ASR，1=后端）

### 常用环境变量

```bash
SKIP_SETUP=1 ./scripts/start_full_stack.sh          # 跳过首次依赖安装
SKIP_TUNNEL=1 ./scripts/start_full_stack.sh         # 不启 cloudflared（仅本机 / Cursor 转发）
BUILD_MODE=cursor ./scripts/start_full_stack.sh     # Cursor 端口转发构建（小包 multipart）
FORCE_RESTART=1 ./scripts/start_full_stack.sh       # 强制重启（杀旧 tmux + 释放 8000/9010）
APPLY_CHINESE_PRESET=1 ./scripts/start_full_stack.sh  # 自动链接中文微调 checkpoint
```

### 停止全部

```bash
cd emotion-agent
./scripts/stop_full_stack.sh
```

Ollama (11434) 默认不停止（通常为后台服务）。

### 浏览器端到端测试

1. 打开脚本输出的 **本机** 或 **公网 HTTPS** 地址，**Ctrl+F5** 硬刷新
2. 确认顶部：**ASR ✓ · LLM ✓ · 模型 checkpoint ✓**
3. 点击「启用摄像头与麦克风」→ 说话 1–3 秒 →「结束采集并推理」
4. 查看 ASR 文本、情绪标签、Ollama 助手回复

### 查看日志与自检

```bash
tmux attach -t emotion-full              # 主服务
tmux attach -t cf_tunnel                 # Cloudflare 隧道
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
./backend/scripts/verify_trained_model.sh
```

---

## 部署后操作（分步 / 高级）

若只需重启后端、或不用 cloudflared，可用以下分步命令：

```bash
# 1. 生产构建（公网 / cloudflared）
cd emotion-agent && ./scripts/build_production.sh

# 2. 仅重启后端+前端（ASR / Ollama 需已在其他终端运行）
FORCE_RESTART=1 ./scripts/start_demo.sh

# 3. tmux 分步启动（不含 cloudflared）
./scripts/start_all_demo.sh

# 4. 单独启动 Cloudflare 隧道（需 8000 已健康）
./scripts/start_cloudflared_tunnel.sh

# 5. 微调完成后切换 preset，再全栈重启
cd ../project && ./scripts/apply_agent_chinese_preset.sh
cd ../emotion-agent
FORCE_RESTART=1 APPLY_CHINESE_PRESET=1 ./scripts/start_full_stack.sh

# 6. 正式公网 nginx + HTTPS 见 deploy/SERVER_DEPLOY.md
#    及 docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md
```

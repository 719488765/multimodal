# 操作指南：中文微调 → 服务启动 → nginx 部署 → 端到端测试

> 适用环境：Linux GPU 服务器（与训练/演示同一台）  
> 项目根目录示例：`/home/lizhichun_24/sda1/code/multimodal`  
> 预计总耗时：微调 2–4h + 部署 30min + 测试 30min

---

## 总览（推荐顺序）

```text
步骤 A  中文微调（GPU，约 2–4h）
步骤 B  切换 agent_chinese preset + 本地验证
步骤 C  生产构建 + 启动 ASR / 后端 / Ollama
步骤 D  nginx + HTTPS（公网域名）
步骤 E  浏览器与命令行测试
```

**不要跳过 A 再测中文准确率**：未微调前 ASR「我很难过」仍易判 neutral。

---

## 步骤 A：中文情绪模型微调

### A.1 环境检查

```bash
conda activate myenv310   # 或你的 Python 3.10+ 环境

# GPU
nvidia-smi

# 预训练 checkpoint（ap2_m1）
ls -lh /home/lizhichun_24/sda1/code/multimodal/project/checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth

# 三数据集 data 目录
ls /home/lizhichun_24/sda1/code/multimodal/project/data/
```

### A.2 首次需联网下载 bert-base-chinese

编辑 `emotion-agent/backend/.env`（微调阶段可临时）或 export：

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
```

### A.3 启动微调（推荐 tmux，SSH 断线不中断）

**推荐：tmux  detached 会话（断线不影响训练）**

```bash
cd /home/lizhichun_24/sda1/code/multimodal
chmod +x run_finetune_chinese_tmux.sh migrate_finetune_to_tmux.sh

# 新启动
./run_finetune_chinese_tmux.sh

# 若已在普通终端跑着、尚未进 Epoch，可迁移到 tmux：
./migrate_finetune_to_tmux.sh
```

常用：

```bash
tmux attach -t finetune_chinese    # 进入看输出；Ctrl+B 再按 D 脱离（训练继续）
tail -f "$(cat project/logs_accuracy_seq/finetune_chinese.latest.log)"
tmux ls                            # 列出会话
```

**注意：必须在 `project/` 目录下执行前台脚本**（不要在 `multimodal/` 根目录直接跑 `./scripts/...`）。

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
chmod +x scripts/finetune_agent_chinese.sh scripts/apply_agent_chinese_preset.sh

# 前台（仅网络稳定时用）
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 ./scripts/finetune_agent_chinese.sh
```

### A.4 成功标志

日志中出现类似：

```text
Skip loading text encoder weight: text_extractor.backbone...
Training samples: 32268, Validation samples: 3723
Epoch 1/12 ...
```

训练结束后存在（路径带时间戳目录）：

```text
checkpoints_accuracy_seq/AP2_M1_chinese_text_agent_YYYYMMDD_HHMMSS/checkpoint_finetune_best_f1.pth
```

### A.5 常见失败

| 现象 | 处理 |
|------|------|
| `word_embeddings 30522 vs 21128` | 已修复 train.py 键前缀；拉最新代码重跑 |
| `Killed` / OOM | 减小 batch 或关闭其他占 GPU 进程 |
| 数据找不到 | 确认 `project/data/` 下 CREMA/MELD/MOSEI 已整理 |

---

## 步骤 B：切换 checkpoint 并本地冒烟

### B.1 链接 checkpoint + 更新 .env

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/apply_agent_chinese_preset.sh
```

该脚本会：

- 将最新 `checkpoint_finetune_best_f1.pth` 链到 `checkpoints_accuracy_seq/AP2_M1_chinese_text_agent/`
- 将 `emotion-agent/backend/.env` 中 `MODEL_CHECKPOINT_PRESET=agent_chinese`

### B.2 确认 .env 关键项

编辑 `emotion-agent/backend/.env`：

```env
MODEL_PROVIDER=current
MODEL_CHECKPOINT_PRESET=agent_chinese
PROJECT_ROOT=/home/lizhichun_24/sda1/code/multimodal/project
MODEL_DEVICE=cuda
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0

ASR_PROVIDER=whisper_api
ASR_WHISPER_API_URL=http://127.0.0.1:9010/v1/audio/transcriptions
ASR_WHISPER_API_LANGUAGE=zh

LLM_PROVIDER=ollama
LLM_API_BASE=http://127.0.0.1:11434
LLM_MODEL=qwen2.5:7b-instruct
```

### B.3 启动依赖服务

**终端 1 — Ollama（若用 LLM）**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent
./scripts/start_ollama.sh
# 或: ollama serve && ollama pull qwen2.5:7b-instruct
```

**终端 2 — ASR**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent/asr-local
./start_server.sh
curl -s http://127.0.0.1:9010/health
```

**终端 3 — 后端 + 前端静态页**

```bash
cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent
./scripts/build_production.sh
FORCE_RESTART=1 ./scripts/start_demo.sh
```

### B.4 命令行健康检查

```bash
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
```

期望：

- `"ok": true`
- `"preset": "agent_chinese"`（或 model 段含 agent_chinese）
- ASR / LLM 状态正常

### B.5 文本推理冒烟（无音视频）

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/emotion/infer-upload \
  -F "session_id=test_zh_1" \
  -F "text=我很难过" \
  -F 'metadata={"source":"cli_test"}' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('label:', d.get('emotion_label'), 'conf:', round(d.get('confidence',0),3))
print('sad prob:', round(d.get('all_probs',[0]*7)[1],3) if d.get('all_probs') else 'N/A')
print('preset:', d.get('checkpoint_preset'), 'source:', d.get('inference_source'))
"
```

期望（微调有效时）：`sad` 概率明显高于 ap2_m1 时期（>0.25 为初步改善信号）。

---

## 步骤 C：生产构建说明

```bash
cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent
./scripts/build_production.sh
```

等价于：

- `VITE_DEPLOY_MODE=server` → 优先整包 multipart，保留 webm
- `VITE_API_BASE_URL=` → 与 nginx 同源

每次改前端后需重新 build，再重启 8000。

---

## 步骤 D'：无 sudo 权限时的替代方案（推荐先看本节）

账户不在 sudoers 时，**无法**执行 `apt install nginx`、写 `/etc/nginx`、申请 certbot 证书。  
**步骤 D 可整段跳过**，用下面三种方式之一完成「外网/浏览器访问 + 稳定上传」。

### 方案 1：Cursor / SSH 端口转发（零权限，已可用）

与之前不同：已启用 **server 模式 + multipart 整包上传**，经 `127.0.0.1:8000` 转发时比 37 次分块稳定得多。

1. 服务器上 backend 已起：`curl -s http://127.0.0.1:8000/api/v1/health`
2. Cursor **Ports** 面板转发 **8000**
3. 本机浏览器打开 **`http://127.0.0.1:8000`**（视为 localhost，麦克风/摄像头可用）

**优点**：无需管理员、无需域名。  
**缺点**：依赖 Cursor/SSH 连接；非真正公网独立部署。

---

### 方案 2：Cloudflare Quick Tunnel（无 sudo、免费 HTTPS）

在用户目录运行 `cloudflared`，把本机 8000 暴露为 `https://xxx.trycloudflare.com`，**无需 root**。

```bash
mkdir -p ~/bin
curl -L -o ~/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/bin/cloudflared

# 确保 backend 在 8000 已启动，另开终端或 tmux：
~/bin/cloudflared tunnel --url http://127.0.0.1:8000
```

终端会打印一行 **`https://xxxx.trycloudflare.com`**，用浏览器打开即可（HTTPS，适合测麦克风）。

**tmux 常驻（断线不断隧道）：**

```bash
tmux new-session -d -s cf_tunnel \
  "$HOME/bin/cloudflared tunnel --url http://127.0.0.1:8000 2>&1 | tee ~/cloudflared_tunnel.log"
grep -o 'https://[^ ]*trycloudflare.com' ~/cloudflared_tunnel.log | tail -1
```

**注意**：Quick Tunnel URL 每次重启会变；仅适合演示/测试，非生产。

---

### 方案 3：内网直连 IP:8000（需管理员开防火墙）

服务器内网 IP 示例：`192.168.1.102`（以 `hostname -I` 为准）。

```bash
# 服务器上 backend 监听 0.0.0.0:8000（start_demo 默认已是）
ss -tlnp | grep 8000
```

同一局域网 PC 浏览器访问：**`http://192.168.1.102:8000`**

需 **机房/管理员** 放行 TCP 8000。  
**限制**：多数浏览器在 **非 localhost 的 HTTP** 下会 **禁止** 摄像头/麦克风；内网 HTTP 可能只能测 API，不能完整采集 demo。

---

### 方案 4：联系管理员安装 nginx（正式公网）

把 `emotion-agent/deploy/nginx.conf` 与 `deploy/SERVER_DEPLOY.md` 发给有 sudo 的管理员，由其在服务器安装 nginx + certbot 并反代 8000。

---

### 无 sudo 时建议怎么选

| 目标 | 推荐 | 上传可靠性 |
|------|------|------------|
| 自己远程调试、要麦克风 | **方案 1** Cursor 转发 8000 + `build_cursor.sh` | 中（单次 ≤384KB multipart） |
| 给外网他人 HTTPS 演示 | **方案 2** cloudflared | 高（HTTPS 直连隧道） |
| 正式域名生产 | **方案 4** 找管理员 nginx | **最高**（稳定大文件） |

### Cursor 隧道 vs 服务器 nginx：对上传的影响（确切结论）

| 对比项 | Cursor `127.0.0.1:8000` 转发 | 服务器 nginx/HTTPS 直连 |
|--------|-------------------------------|-------------------------|
| HTTP 请求次数 | 以前 37+ 次易失败；**现已改为 1 次 multipart** | **1 次** multipart |
| 瓶颈 | **Cursor 隧道**（延迟、断连、大 POST 超时） | 带宽与 nginx 配置（20MB 上限） |
| 典型 payload | 音频 ~96KB + 视频 JPEG/webm，**合计 ≤384KB** | 音频 + webm **≤5MB** |
| 上传失败主因 | **不是服务器算力**，是隧道不适合多请求/大包 | 极少失败 |
| 视频质量 | 隧道下 webm 小或 JPEG 单帧 | 可保留完整 webm 4 帧 |
| 结论 | **能用但不如直连稳定**；已针对隧道优化 | **推荐正式演示与测准确率** |

**构建命令（按访问方式二选一）：**

```bash
cd emotion-agent
./scripts/build_cursor.sh          # Cursor 转发 127.0.0.1:8000
./scripts/build_production.sh      # nginx / cloudflared / 公网 HTTPS
FORCE_RESTART=1 ./scripts/start_demo.sh
```

浏览器运行日志应显示：`上传策略: Cursor 隧道（单次小包 multipart，禁止分块）` 或 `服务器直连…`。

---

## 步骤 D：nginx + HTTPS 公网部署（需 sudo，无权限则跳过改用上节 D'）

### D.1 安装 nginx 与 certbot（Ubuntu）

```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
sudo systemctl enable nginx
```

> 若提示 `not in the sudoers file`，请改用上文 **步骤 D'**。

### D.2 准备域名

- 将域名 A 记录指向服务器公网 IP
- 假设域名为 `emotion.example.com`（下文替换为你的域名）

### D.3 编辑 nginx 配置

```bash
sudo cp /home/lizhichun_24/sda1/code/multimodal/emotion-agent/deploy/nginx.conf \
  /etc/nginx/sites-available/emotion-agent

sudo sed -i 's/YOUR_DOMAIN/emotion.example.com/g' /etc/nginx/sites-available/emotion-agent

sudo ln -sf /etc/nginx/sites-available/emotion-agent /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default   # 可选，避免冲突
```

**首次无证书时**：先注释 `nginx.conf` 中 `listen 443 ssl` 整段和 `return 301`，只保留 `listen 80` + `location /` 反代到 8000，用于 certbot 申请证书。

简化 HTTP-only 调试片段（临时）：

```nginx
server {
    listen 80;
    server_name emotion.example.com;
    client_max_body_size 20m;
    proxy_read_timeout 600s;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### D.4 防火墙

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
# 不要对公网开放 8000、9010、11434
```

### D.5 启动后端（仅本机 8000）

确认步骤 B 中 ASR、Ollama、backend 已在运行。

### D.6 测试并重载 nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### D.7 申请 HTTPS 证书

```bash
sudo certbot --nginx -d emotion.example.com
```

按提示选择重定向 HTTP→HTTPS。

### D.8 更新 CORS

`emotion-agent/backend/.env`：

```env
CORS_ALLOW_ORIGIN=https://emotion.example.com
APP_ENV=prod
```

重启后端：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent
FORCE_RESTART=1 ./scripts/start_demo.sh
```

### D.9 验证 HTTPS

```bash
curl -s https://emotion.example.com/api/v1/health | python3 -m json.tool
```

---

## 步骤 E：端到端测试清单

### E.1 命令行

| 项 | 命令 | 期望 |
|----|------|------|
| 健康 | `curl -s https://你的域名/api/v1/health` | ok:true, preset:agent_chinese |
| 文本 | 见 B.5，域名改为 https | sad 概率提升 |
| 批量分块 API | localhost 测 `/api/v1/emotion/upload-chunks-batch` | accepted:true |

### E.2 浏览器（推荐 Chrome）

1. 打开 `https://你的域名/`（**不要用** Cursor 转发的 127.0.0.1:8000）
2. 允许摄像头、麦克风（HTTPS 下权限更稳定）
3. 点「开始录制」→ 说 **「我很难过」** 约 5–30 秒（Cloudflare 隧道约 10 秒）→「结束采集并推理」
   - 系统将完整音频按 **3 秒一窗** 多次送入情绪模型，**近端加权** 得到最终标签，并展示情绪时间线

**运行日志应出现：**

- `服务器模式：保留 webm (...KB...)`
- `上传 multipart 1/1` 或少量分块（非 37 块）
- **不应**出现「改用 JPEG」（除非录制失败）

**流水线监控（pipeline_trace）：**

| 阶段 | 期望 |
|------|------|
| 1_ingest | video_bytes > 0 |
| 2_asr | text 含「难过」 |
| 4_emotion_model | inference_source=checkpoint, preset=agent_chinese |
| video | decode_mode=**video_file**, frames_extracted=**4** |
| text | tokenizer=**bert-base-chinese** |

**情感结果：**

- Top1 倾向 **sad** 或 **fear**（不必 100%，但 sad 概率应 > ap2_m1 时 ~0.13）
- LLM 回复依据中引用模型概率，而非仅 ASR

### E.3 对比测试（可选）

切换回 ap2_m1 再测同一句，对比 sad 概率：

```bash
# .env 改 MODEL_CHECKPOINT_PRESET=ap2_m1 后 FORCE_RESTART=1 ./scripts/start_demo.sh
```

### E.4 失败排查

| 现象 | 排查 |
|------|------|
| 502 Bad Gateway | backend 8000 未启动；`curl localhost:8000/api/v1/health` |
| Failed to fetch | 是否仍用 Cursor 转发；应直接访问 https 域名 |
| 仍 JPEG 回退 | 确认 build 时 VITE_DEPLOY_MODE=server；硬刷新 Ctrl+F5 |
| ASR 空 | `curl localhost:9010/health`；重启 asr-local |
| 仍 neutral | 试切换 **ap2_m1** 对比；确认 ASR 含「高兴/哈哈」时会触发 **5_asr_calibration**；多帧 `video_decode_mode=multi_frame_sequence` |
| 开心被判平静 | 流水线 stage **5_asr_calibration** 应为 applied；或缩短录制走单窗 + 重训 v2 |
| LLM 无回复 | `curl localhost:11434/api/tags`；ollama 是否运行 |

---

## 步骤 F：长期运行（可选 systemd）

编辑 `emotion-agent/deploy/systemd/*.service` 中的路径与用户名后：

```bash
sudo cp emotion-agent/deploy/systemd/emotion-asr.service /etc/systemd/system/
sudo cp emotion-agent/deploy/systemd/emotion-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now emotion-asr emotion-backend
```

Ollama 需单独配置服务或使用 `start_all_demo.sh` tmux 方案。

---

## 步骤 G：准确率优化（审计文档 + 单域 MELD + 仲裁）

**主文档**：[`project/docs/EMOTION_ACCURACY_AUDIT_AND_ROADMAP.md`](../project/docs/EMOTION_ACCURACY_AUDIT_AND_ROADMAP.md)

**推荐部署（无需重训）：**

```bash
cd project
chmod +x scripts/apply_deploy_preset.sh
./scripts/apply_deploy_preset.sh meld_only
cd ../emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh
```

**已内置：**

1. 默认权重 **meld_only**（MELD 单域，mp4 与在线 ResNet 一致；优于三混合 OOD）
2. 流水线 **5_asr_calibration** + **6_arbitration**（纯「哈哈」+ 扁平概率 → happy）
3. 页面可选 preset：meld_only / ap2_m1 / mosei_only / agent_chinese
4. 多帧采集 + 时序单窗短采集策略

**回归：**

```bash
cd project
python3 scripts/eval_agent_capture_cases.py
python3 -m pytest tests/test_asr_emotion_calibration.py -q
cd ../emotion-agent/backend && python3 -m pytest tests/test_emotion_arbitration.py -q
```

**单域重训（GPU，进一步提升）：**

```bash
cd project && HF_HUB_OFFLINE=0 ./scripts/train_meld_agent.sh
# 完成后将 best ckpt 链到 preset 并 apply_deploy_preset.sh meld_only
```

---

## 快速命令备忘

```bash
# 微调
cd project && HF_HUB_OFFLINE=0 ./scripts/finetune_agent_chinese.sh
cd project && HF_HUB_OFFLINE=0 ./scripts/finetune_agent_chinese_v2.sh

# 切换 preset
cd project && ./scripts/apply_agent_chinese_preset.sh

# 生产 build + 启动
cd emotion-agent && ./scripts/build_production.sh && FORCE_RESTART=1 ./scripts/start_demo.sh

# 健康
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
```

---

## 相关文档

- `deploy/SERVER_DEPLOY.md` — 部署摘要
- `START_DEMO.txt` — 本地演示
- `docs/ARCHITECTURE.md` — 架构与数据流
- `project/docs/WEEKLY_REPORT_20260523_0530.txt` — 本周工作与已知问题

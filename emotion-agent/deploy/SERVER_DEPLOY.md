# 公网服务器部署（nginx + HTTPS）

## 架构

浏览器 `https://YOUR_DOMAIN` → nginx:443 → uvicorn:8000（静态 `frontend/dist` + `/api`）  
ASR 仅监听 `127.0.0.1:9010`；Ollama 仅 `127.0.0.1:11434`。

与 Cursor 开发区别：**无端口转发**，前端 `VITE_DEPLOY_MODE=server` 优先单次 multipart 上传 webm，避免 JPEG 回退。

## 1. 构建与配置

```bash
cd emotion-agent
cp deploy/.env.production.example backend/.env
# 编辑 backend/.env：PROJECT_ROOT、CORS_ALLOW_ORIGIN、MODEL_CHECKPOINT_PRESET

chmod +x scripts/build_production.sh
./scripts/build_production.sh
```

## 2. 中文情绪模型微调（首次部署前）

```bash
cd ../project
./scripts/finetune_agent_chinese.sh
# 完成后 backend/.env 设 MODEL_CHECKPOINT_PRESET=agent_chinese
```

## 3. 启动服务

```bash
# ASR
cd emotion-agent/asr-local && ./start_server.sh

# 后端 + 静态页（或 systemd emotion-backend.service）
cd emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh
```

## 4. nginx + TLS

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/emotion-agent
# 替换 YOUR_DOMAIN
sudo ln -sf /etc/nginx/sites-available/emotion-agent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d YOUR_DOMAIN
```

## 5. systemd（可选）

编辑 `deploy/systemd/*.service` 中的路径与 `YOUR_USER`，然后：

```bash
sudo cp deploy/systemd/emotion-asr.service /etc/systemd/system/
sudo cp deploy/systemd/emotion-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now emotion-asr emotion-backend
```

## 6. 验收

- `curl -s https://YOUR_DOMAIN/api/v1/health` → `{"ok":true,...}`
- 浏览器采集后 pipeline：`decode_mode=video_file`，`frames_extracted=4`
- 运行日志无「改用 JPEG」（除非录制失败）

## 防火墙

- 开放：443（及 80 用于 certbot 重定向）
- 不对外开放：8000、9010、11434（仅本机 + nginx 反代）

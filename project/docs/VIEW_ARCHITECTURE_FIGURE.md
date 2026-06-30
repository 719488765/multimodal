# 如何在本地浏览器查看远程架构图 HTML

架构图文件在**远程服务器**上，本地不能直接打开 `file:///home/...` 路径。任选下面一种方式即可。

---

## 方式一：Cursor 端口转发（推荐，已在用 Remote SSH 时）

1. 在 **远程终端** 启动静态服务：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
chmod +x scripts/serve_architecture_figure.sh
./scripts/serve_architecture_figure.sh
```

2. 在 Cursor 底部打开 **Ports（端口）** 面板。
3. 点击 **Forward a Port**，输入 `8765`（若已自动出现则跳过）。
4. 在 8765 一行点 **Open in Browser**，或本地浏览器访问：

```
http://127.0.0.1:8765/system_architecture_figure.html
```

> 服务只绑定 `127.0.0.1`，外网无法直接访问，必须通过 SSH/Cursor 转发，更安全。

---

## 方式二：本地 SSH 隧道（不用 Cursor 时）

在**你本地电脑**的终端执行（把 `USER`、`REMOTE_HOST` 换成实际账号和服务器地址）：

```bash
ssh -L 8765:127.0.0.1:8765 USER@REMOTE_HOST
```

保持该 SSH 连接不断开；远程另开终端运行 `./scripts/serve_architecture_figure.sh`。

本地浏览器打开：

```
http://127.0.0.1:8765/system_architecture_figure.html
```

---

## 方式三：通过 emotion-agent 后端（8000 已启动时）

若已运行 `emotion-agent/scripts/start_demo.sh` 或 nginx 反代到 8000，可直接访问：

```
http://127.0.0.1:8000/docs/system-architecture
```

（需对 8000 做同样的端口转发或 SSH 隧道。）

公网域名部署时：

```
https://YOUR_DOMAIN/docs/system-architecture
```

---

## 方式四：下载到本地直接打开

在**本地电脑**执行：

```bash
scp USER@REMOTE_HOST:/home/lizhichun_24/sda1/code/multimodal/project/docs/figures/system_architecture_figure.html ~/Downloads/
```

然后双击 `~/Downloads/system_architecture_figure.html` 用浏览器打开（需联网加载 Google Fonts）。

---

## 常见问题

| 问题 | 处理 |
|------|------|
| 端口 8765 被占用 | `ARCH_FIGURE_PORT=8766 ./scripts/serve_architecture_figure.sh`，转发 8766 |
| Cursor 看不到 Ports | 命令面板 `Forward a Port` 或 View → Ports |
| 页面空白/字体异常 | 检查网络能否访问 fonts.googleapis.com，或改用方式四下载本地打开 |
| 修改 HTML 后看不到更新 | 刷新浏览器（Ctrl+F5）；静态服务无需重启 |

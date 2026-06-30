#!/usr/bin/env bash
# 安装 cloudflared 到 ~/bin（多镜像，无需 sudo）
set -euo pipefail

BIN="${HOME}/bin/cloudflared"
VER="${CLOUDFLARED_VERSION:-2024.12.2}"
mkdir -p "$(dirname "$BIN")"

if [[ -x "$BIN" ]]; then
  echo "已存在: $BIN"
  "$BIN" --version 2>/dev/null || true
  exit 0
fi

URLS=(
  "https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/download/${VER}/cloudflared-linux-amd64"
  "https://mirror.ghproxy.com/https://github.com/cloudflare/cloudflared/releases/download/${VER}/cloudflared-linux-amd64"
  "https://gh-proxy.com/https://github.com/cloudflare/cloudflared/releases/download/${VER}/cloudflared-linux-amd64"
  "https://github.com/cloudflare/cloudflared/releases/download/${VER}/cloudflared-linux-amd64"
)

download_one() {
  local url="$1"
  local tmp="${BIN}.part"
  echo "==> 尝试: $url"
  rm -f "$tmp"
  if curl -fL --connect-timeout 20 --max-time 600 --retry 2 --retry-delay 3 \
    -o "$tmp" "$url"; then
    :
  elif command -v wget >/dev/null 2>&1; then
    wget -T 30 --tries=2 -O "$tmp" "$url" || return 1
  else
    return 1
  fi
  local size
  size=$(wc -c < "$tmp" 2>/dev/null || echo 0)
  if [[ "$size" -lt 10000000 ]]; then
    echo "    文件过小 (${size} bytes)，跳过" >&2
    rm -f "$tmp"
    return 1
  fi
  if ! file "$tmp" 2>/dev/null | grep -qE 'ELF|executable'; then
    echo "    非可执行 ELF，跳过" >&2
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$BIN"
  chmod +x "$BIN"
  return 0
}

for u in "${URLS[@]}"; do
  if download_one "$u"; then
    echo "==> 安装成功: $BIN"
    "$BIN" --version
    exit 0
  fi
done

cat >&2 <<'EOF'

ERROR: 无法从 GitHub/镜像下载 cloudflared（服务器出网受限）。

【手动安装】在 Windows 本机下载后 scp 上传:
  https://github.com/cloudflare/cloudflared/releases/download/2024.12.2/cloudflared-linux-amd64
  scp cloudflared-linux-amd64 lizhichun_24@<服务器IP>:~/bin/cloudflared
  chmod +x ~/bin/cloudflared && ~/bin/cloudflared --version

【备选】Windows OpenSSH 转发（常比 Cursor 稳）:
  ssh -L 8000:127.0.0.1:8000 lizhichun_24@<服务器IP>
  浏览器 http://127.0.0.1:8000 + build_cursor.sh

EOF
exit 1

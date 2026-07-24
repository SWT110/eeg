#!/usr/bin/env bash
set -Eeuo pipefail

SSH_USER="root"
SSH_HOST="38.244.62.168"
SSH_KEY="$HOME/.ssh/pi_accel_ed25519"

SOCKS_PORT="31880"
HTTP_PORT="31881"

RUNTIME_DIR="$HOME/.cache/pi-proxy"
CONTROL_SOCKET="$RUNTIME_DIR/ssh-control.sock"
HPTS_PID_FILE="$RUNTIME_DIR/hpts.pid"

port_listening() {
  local port="$1"
  python3 - "$port" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

echo "[1/3] 关闭 HTTP 转发代理"

if [ -f "$HPTS_PID_FILE" ]; then
  HPTS_PID="$(cat "$HPTS_PID_FILE" || true)"
  if [ -n "${HPTS_PID:-}" ] && kill -0 "$HPTS_PID" 2>/dev/null; then
    kill "$HPTS_PID" || true
  fi
fi

pkill -f "http-proxy-to-socks.*${HTTP_PORT}" 2>/dev/null || true
pkill -f "http-proxy-to-socks.*${SOCKS_PORT}" 2>/dev/null || true

echo "[2/3] 关闭 SSH SOCKS5 代理"

if [ -S "$CONTROL_SOCKET" ]; then
  ssh -S "$CONTROL_SOCKET" -O exit \
    -i "$SSH_KEY" \
    "$SSH_USER@$SSH_HOST" 2>/dev/null || true
fi

pkill -f "ssh.*-D.*127.0.0.1:${SOCKS_PORT}.*${SSH_USER}@${SSH_HOST}" 2>/dev/null || true
pkill -f "ssh.*127.0.0.1:${SOCKS_PORT}" 2>/dev/null || true

echo "[3/3] 检查端口"

if port_listening "$SOCKS_PORT" || port_listening "$HTTP_PORT"; then
  echo "仍有进程占用 $SOCKS_PORT 或 $HTTP_PORT，请手动检查。"
else
  echo "已关闭 $SOCKS_PORT 和 $HTTP_PORT。"
fi

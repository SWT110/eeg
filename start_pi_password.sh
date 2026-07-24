#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$HOME/SWT/work/eeg"

SSH_USER="root"
SSH_HOST="38.244.62.168"

SOCKS_HOST="127.0.0.1"
SOCKS_PORT="31880"

HTTP_HOST="127.0.0.1"
HTTP_PORT="31881"

RUNTIME_DIR="$HOME/.cache/pi-proxy"
CONTROL_SOCKET="$RUNTIME_DIR/ssh-control.sock"
HPTS_PID_FILE="$RUNTIME_DIR/hpts.pid"
HPTS_LOG="/tmp/hpts-31881.log"

mkdir -p "$RUNTIME_DIR"

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

echo "[1/5] 进入项目目录：$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "[2/5] 启动 SSH SOCKS5 代理：$SOCKS_HOST:$SOCKS_PORT"

if port_listening "$SOCKS_PORT"; then
  echo "SOCKS5 端口 $SOCKS_PORT 已经在监听，跳过 SSH 启动。"
else
  # 如果上次异常退出留下了坏的 control socket，先清掉，避免复用失败。
  if [[ -S "$CONTROL_SOCKET" ]]; then
    if ! ssh -S "$CONTROL_SOCKET" -O check -o BatchMode=yes "$SSH_USER@$SSH_HOST" >/dev/null 2>&1; then
      rm -f "$CONTROL_SOCKET"
    fi
  fi

  echo "将连接到 $SSH_USER@$SSH_HOST。请在下面输入服务器登录密码。"

  ssh -M -S "$CONTROL_SOCKET" -fN \
    -D "$SOCKS_HOST:$SOCKS_PORT" \
    -o PreferredAuthentications=password,keyboard-interactive \
    -o PubkeyAuthentication=no \
    -o PasswordAuthentication=yes \
    -o BatchMode=no \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "$SSH_USER@$SSH_HOST"
fi

echo "[3/5] 启动 SOCKS5 -> HTTP 代理：$HTTP_HOST:$HTTP_PORT"

if port_listening "$HTTP_PORT"; then
  echo "HTTP 代理端口 $HTTP_PORT 已经在监听，跳过 hpts 启动。"
else
  nohup npx -y http-proxy-to-socks \
    -s "$SOCKS_HOST:$SOCKS_PORT" \
    -p "$HTTP_PORT" \
    -l "$HTTP_HOST" \
    > "$HPTS_LOG" 2>&1 &

  echo $! > "$HPTS_PID_FILE"

  for i in $(seq 1 30); do
    if port_listening "$HTTP_PORT"; then
      break
    fi
    sleep 1
  done
fi

echo "[4/5] 测试 HTTP 代理"

if curl -fsSL -x "http://$HTTP_HOST:$HTTP_PORT" https://registry.npmjs.org/ >/dev/null; then
  echo "HTTP 代理测试成功。"
else
  echo "HTTP 代理测试失败。查看日志：$HPTS_LOG"
  tail -n 30 "$HPTS_LOG" || true
  exit 1
fi

echo "[5/5] 启动 Pi Agent"
echo "项目目录：$(pwd)"
echo "代理地址：http://$HTTP_HOST:$HTTP_PORT"
echo

env \
  -u ALL_PROXY \
  -u all_proxy \
  HTTP_PROXY="http://$HTTP_HOST:$HTTP_PORT" \
  http_proxy="http://$HTTP_HOST:$HTTP_PORT" \
  HTTPS_PROXY="http://$HTTP_HOST:$HTTP_PORT" \
  https_proxy="http://$HTTP_HOST:$HTTP_PORT" \
  npm_config_proxy="http://$HTTP_HOST:$HTTP_PORT" \
  npm_config_https_proxy="http://$HTTP_HOST:$HTTP_PORT" \
  NO_PROXY="127.0.0.1,localhost,::1" \
  no_proxy="127.0.0.1,localhost,::1" \
  npx -y --package @earendil-works/pi-coding-agent pi "$@"

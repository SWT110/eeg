#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$HOME/SWT/work/eeg"

SSH_USER="root"
SSH_HOST="84.75.220.173"
SSH_KEY="$HOME/.ssh/pi_accel_ed25519"

SOCKS_HOST="127.0.0.1"
SOCKS_PORT="31880"

HTTP_HOST="127.0.0.1"
HTTP_PORT="31881"

RUNTIME_DIR="$HOME/.cache/pi-proxy"
CONTROL_SOCKET="$RUNTIME_DIR/ssh-control.sock"
HPTS_PID_FILE="$RUNTIME_DIR/hpts.pid"
HPTS_LOG="/tmp/hpts-31881.log"

mkdir -p "$RUNTIME_DIR"

# SSH/PC 远程登录时 PATH 可能没有加载 conda/nvm，导致找不到 npx。
NPX_BIN="${NPX_BIN:-}"
if [[ -z "$NPX_BIN" ]]; then
  for candidate in \
    "$HOME/miniconda3/envs/ps-mt/bin/npx" \
    "/usr/local/nvm/versions/node/v16.15.1/bin/npx" \
    "$HOME/miniconda3/bin/npx"; do
    if [[ -x "$candidate" ]]; then
      NPX_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$NPX_BIN" ]]; then
  NPX_BIN="$(command -v npx || true)"
fi
if [[ -z "$NPX_BIN" ]]; then
  echo "错误：找不到 npx。请先安装/加载 Node.js，或设置 NPX_BIN=/path/to/npx"
  exit 1
fi
# npx 的 shebang 通常是 /usr/bin/env node，所以也要把 node 所在目录加入 PATH。
export PATH="$(dirname "$NPX_BIN"):$PATH"

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
  ssh -M -S "$CONTROL_SOCKET" -fN \
    -i "$SSH_KEY" \
    -D "$SOCKS_HOST:$SOCKS_PORT" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "$SSH_USER@$SSH_HOST"
fi

echo "[3/5] 启动 SOCKS5 -> HTTP 代理：$HTTP_HOST:$HTTP_PORT"

if port_listening "$HTTP_PORT"; then
  echo "HTTP 代理端口 $HTTP_PORT 已经在监听，跳过 hpts 启动。"
else
  nohup "$NPX_BIN" -y http-proxy-to-socks \
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
  "$NPX_BIN" -y --package @earendil-works/pi-coding-agent pi "$@"

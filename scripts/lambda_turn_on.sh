#!/usr/bin/env bash
# One-command turn-on: provision Lambda GPU, tunnel, idle watchdog, and chat UI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
RUNTIME_DIR="${ROOT}/.lambda-runtime"
PYTHON="${PYTHON:-python3}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${LAMBDA_CLOUD_API_KEY:?Set LAMBDA_CLOUD_API_KEY in .env}"
: "${MINISGL_API_KEY:?Set MINISGL_API_KEY in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"

PORT="${MINISGL_PORT:-1919}"
SSH_KEY="${SSH_PRIVATE_KEY_PATH:-${HOME}/.ssh/id_ed25519}"
CHAT_LISTEN="${MINISGL_CHAT_LISTEN:-127.0.0.1:8765}"
IDLE_TIMEOUT_S="${MINISGL_IDLE_TIMEOUT_S:-3600}"
DEPLOY_ARGS=()

mkdir -p "${RUNTIME_DIR}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      DEPLOY_ARGS+=("--smoke")
      shift
      ;;
    --reuse)
      DEPLOY_ARGS+=("--reuse")
      shift
      ;;
    --no-browser)
      NO_BROWSER=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--smoke] [--reuse] [--no-browser]" >&2
      exit 2
      ;;
  esac
done

stop_if_running() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
}

export ROOT

echo "==> Checking Lambda instance and inference server…"
set +e
HEALTH_OK="$("${PYTHON}" - <<'PY'
import os
import sys

ROOT = os.environ["ROOT"]
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lambda_common import load_env_file, fetch_health, find_instance_for_ip, list_reusable_instances, require, ENV_FILE

load_env_file(ENV_FILE)
token = require("LAMBDA_CLOUD_API_KEY")
ip = os.getenv("LAMBDA_PUBLIC_IP", "")
instance = find_instance_for_ip(token, ip) or (list_reusable_instances(token)[0] if list_reusable_instances(token) else None)
healthy = fetch_health()
print("1" if healthy else "0")
print(instance["id"] if instance else "")
PY
)"
set -e
HEALTH_STATUS="$(echo "${HEALTH_OK}" | sed -n '1p')"
INSTANCE_ID="$(echo "${HEALTH_OK}" | sed -n '2p')"

if [[ "${HEALTH_STATUS}" != "1" ]]; then
  echo "==> Inference server not reachable — deploying…"
  if [[ -n "${INSTANCE_ID}" ]]; then
    DEPLOY_ARGS+=("--reuse")
  fi
  "${PYTHON}" "${ROOT}/scripts/lambda_deploy.py" "${DEPLOY_ARGS[@]}"
else
  echo "==> Inference server already healthy."
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a
: "${LAMBDA_PUBLIC_IP:?LAMBDA_PUBLIC_IP missing after deploy}"

stop_if_running "${RUNTIME_DIR}/tunnel.pid"
stop_if_running "${RUNTIME_DIR}/watchdog.pid"

echo "==> Starting SSH tunnel (${PORT} -> ${LAMBDA_PUBLIC_IP})…"
nohup ssh -N \
  -i "${SSH_KEY}" \
  -o StrictHostKeyChecking=accept-new \
  -L "${PORT}:127.0.0.1:${PORT}" \
  "ubuntu@${LAMBDA_PUBLIC_IP}" \
  > "${RUNTIME_DIR}/tunnel.log" 2>&1 &
echo $! > "${RUNTIME_DIR}/tunnel.pid"

echo "==> Waiting for /health…"
"${PYTHON}" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts"))
from lambda_common import wait_for_health
wait_for_health(timeout_s=120, interval_s=3)
print("Mini-SGLang is healthy.")
PY

echo "==> Starting idle watchdog (${IDLE_TIMEOUT_S}s timeout)…"
nohup "${PYTHON}" "${ROOT}/scripts/lambda_idle_watchdog.py" \
  --idle-timeout-s "${IDLE_TIMEOUT_S}" \
  > "${RUNTIME_DIR}/watchdog.log" 2>&1 &
echo $! > "${RUNTIME_DIR}/watchdog.pid"

CHAT_HOST="${CHAT_LISTEN%%:*}"
CHAT_PORT="${CHAT_LISTEN##*:}"
CHAT_URL="http://${CHAT_HOST}:${CHAT_PORT}/"

echo ""
echo "=============================================="
echo " Mini-SGLang Lambda is ready"
echo "=============================================="
echo " Chat UI:     ${CHAT_URL}"
echo " API (local): http://127.0.0.1:${PORT}/v1"
echo " Instance IP: ${LAMBDA_PUBLIC_IP}"
echo " Idle cutoff: ${IDLE_TIMEOUT_S}s without inference"
echo ""
echo " Turn off:    python scripts/lambda_turn_off.py"
echo " Tunnel log:  ${RUNTIME_DIR}/tunnel.log"
echo " Watchdog:    ${RUNTIME_DIR}/watchdog.log"
echo "=============================================="
echo ""

if [[ -z "${NO_BROWSER:-}" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "${CHAT_URL}" || true
  fi
fi

echo "Starting chat UI (Ctrl+C stops the UI only; tunnel/watchdog keep running)…"
exec "${PYTHON}" "${ROOT}/scripts/serve_lambda_chat.py" --listen "${CHAT_LISTEN}"

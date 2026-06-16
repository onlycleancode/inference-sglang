#!/usr/bin/env bash
# Open an SSH tunnel from your Mac to Mini-SGLang on the Lambda VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

: "${LAMBDA_PUBLIC_IP:?Set LAMBDA_PUBLIC_IP in .env after deploy}"
: "${SSH_PRIVATE_KEY_PATH:=${HOME}/.ssh/id_ed25519}"
PORT="${MINISGL_PORT:-1919}"

echo "Tunneling localhost:${PORT} -> ${LAMBDA_PUBLIC_IP}:${PORT}"
echo "Leave this running in a separate terminal, then call the API at http://127.0.0.1:${PORT}/v1"
exec ssh -N \
  -i "${SSH_PRIVATE_KEY_PATH}" \
  -o StrictHostKeyChecking=accept-new \
  -L "${PORT}:127.0.0.1:${PORT}" \
  "ubuntu@${LAMBDA_PUBLIC_IP}"

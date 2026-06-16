#!/usr/bin/env bash
# Build a deployable tarball for Lambda GPU VMs (no secrets).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${ROOT}/dist"
ARCHIVE="${OUT_DIR}/minisgl-lambda-${VERSION}.tar.gz"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

FILES=(
  Dockerfile
  .dockerignore
  pyproject.toml
  README.md
  python
  deploy/lambda
  scripts/minisgl_remote_infer.py
  scripts/minisgl_smoke_check.py
  scripts/lambda_deploy.py
  scripts/lambda_tunnel.sh
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Would create ${ARCHIVE} with:"
  printf '  %s\n' "${FILES[@]}"
  exit 0
fi

mkdir -p "${OUT_DIR}"

tar -czf "${ARCHIVE}" \
  --exclude='**/__pycache__' \
  --exclude='**/*.pyc' \
  --exclude='.env' \
  -C "${ROOT}" \
  "${FILES[@]}"

echo "Created ${ARCHIVE}"
tar -tzf "${ARCHIVE}" | head -20
echo "..."

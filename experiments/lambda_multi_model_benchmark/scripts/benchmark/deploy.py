"""Deploy MiniSGLang to benchmark nodes with config parity."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from benchmark.config import REMOTE_PORT, SERVER_ARGS


def _run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def package_repo() -> Path:
    _run([str(ROOT / "scripts" / "package_lambda.sh")])
    archives = sorted((ROOT / "dist").glob("minisgl-lambda-*.tar.gz"), reverse=True)
    if not archives:
        raise RuntimeError("Package tarball not found after packaging.")
    return archives[0]


def wait_for_ssh(ip: str, ssh_key: Path, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = subprocess.run(
            [
                "ssh",
                "-i",
                str(ssh_key),
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=5",
                f"ubuntu@{ip}",
                "echo ready",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for SSH on {ip}")


def git_sha() -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def remote_deploy_node(
    *,
    ip: str,
    ssh_key: Path,
    archive: Path,
    model_id: str,
    api_key: str,
    hf_token: str,
    port: int = REMOTE_PORT,
) -> float:
    """Deploy one node and return model load duration in seconds."""
    scp_cmd = [
        "scp",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(archive),
        f"ubuntu@{ip}:~/minisgl-lambda.tar.gz",
    ]
    _run(scp_cmd)

    server_args = " ".join(SERVER_ARGS)
    remote_env = {
        "MINISGL_MODEL": model_id,
        "MINISGL_PORT": str(port),
        "MINISGL_API_KEY": api_key,
        "MINISGL_REQUIRE_API_KEY": "1",
        "HF_TOKEN": hf_token,
        "HF_HOME": "/app/.cache/huggingface",
        "TVM_FFI_CACHE_DIR": "/app/.cache/tvm-ffi",
        "FLASHINFER_WORKSPACE_BASE": "/app/.cache/flashinfer",
        "MINISGL_DTYPE": "bfloat16",
    }
    env_body = "\n".join(f"{key}={value}" for key, value in remote_env.items())

    remote_script = f"""set -euo pipefail
mkdir -p ~/minisgl && cd ~/minisgl
tar -xzf ~/minisgl-lambda.tar.gz
cd deploy/lambda
cat > .env <<'ENV'
{env_body}
ENV
cat > benchmark-compose.override.yml <<'OVERRIDE'
services:
  minisgl:
    command:
      - --model
      - ${{MINISGL_MODEL}}
      - --host
      - 0.0.0.0
      - --port
      - "{port}"
      - --dtype
      - bfloat16
      - --cache-type
      - radix
      - --page-size
      - "64"
      - --max-seq-len-override
      - "4096"
      - --cuda-graph-max-bs
      - "64"
OVERRIDE
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker.io
  sudo usermod -aG docker ubuntu || true
fi
if ! docker compose version >/dev/null 2>&1; then
  sudo apt-get install -y -qq docker-compose-plugin || true
fi
START=$(date +%s)
sudo docker compose -f compose.yml -f benchmark-compose.override.yml up -d --build --force-recreate
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:{port}/health >/dev/null; then
    END=$(date +%s)
    echo "MODEL_LOAD_SECONDS=$((END - START))"
    exit 0
  fi
  sleep 10
done
echo "Timed out waiting for Mini-SGLang health endpoint" >&2
sudo docker logs lambda-minisgl-1 2>&1 | tail -40
exit 1
"""

    load_start = time.perf_counter()
    ssh_cmd = [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"ubuntu@{ip}",
        "bash",
        "-s",
    ]
    proc = subprocess.run(ssh_cmd, input=remote_script.encode(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Remote deploy failed on {ip}: {proc.stderr or proc.stdout}")

    model_load_s = time.perf_counter() - load_start
    for line in proc.stdout.splitlines():
        if line.startswith("MODEL_LOAD_SECONDS="):
            model_load_s = float(line.split("=", 1)[1])
    return model_load_s

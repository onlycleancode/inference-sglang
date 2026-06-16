#!/usr/bin/env python3
"""Provision a Lambda GPU instance and deploy Mini-SGLang."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lambda_common import (  # noqa: E402
    ENV_FILE,
    api_request,
    launch_instance,
    list_reusable_instances,
    load_env_file,
    require,
    resolve_instance_types,
    update_env_ip,
)


def read_public_key(private_key_path: Path) -> str:
    pub_path = Path(str(private_key_path) + ".pub")
    if not pub_path.exists():
        raise SystemExit(f"Missing public key: {pub_path}")
    return pub_path.read_text().strip()


def find_ssh_key_name(token: str, public_key: str) -> str:
    keys = api_request("GET", "/ssh-keys", token).get("data", [])
    for item in keys:
        if item.get("public_key", "").strip() == public_key:
            return item["name"]
    raise SystemExit(
        "SSH public key is not registered in Lambda. Upload it in the Lambda console first."
    )


def wait_for_ip(token: str, instance_id: str, timeout_s: int = 600) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        instances = api_request("GET", "/instances", token).get("data", [])
        for item in instances:
            if item.get("id") == instance_id:
                ip = item.get("ip")
                if ip:
                    return ip
        time.sleep(10)
    raise SystemExit(f"Timed out waiting for IP on instance {instance_id}")


def run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def package() -> Path:
    run([str(ROOT / "scripts" / "package_lambda.sh")])
    archives = sorted((ROOT / "dist").glob("minisgl-lambda-*.tar.gz"), reverse=True)
    if not archives:
        raise SystemExit("Package tarball not found after packaging.")
    return archives[0]


def wait_for_ssh(ip: str, ssh_key: Path, timeout_s: int = 600) -> None:
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
            print("SSH is ready")
            return
        time.sleep(10)
    raise SystemExit(f"Timed out waiting for SSH on {ip}")


def remote_deploy(
    ip: str,
    ssh_key: Path,
    archive: Path,
    *,
    model: str,
    smoke_model: str,
    port: str,
    api_key: str,
    hf_token: str,
) -> None:
    scp_cmd = [
        "scp",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(archive),
        f"ubuntu@{ip}:~/minisgl-lambda.tar.gz",
    ]
    run(scp_cmd)

    remote_env = {
        "MINISGL_MODEL": model,
        "MINISGL_SMOKE_MODEL": smoke_model,
        "MINISGL_PORT": port,
        "MINISGL_API_KEY": api_key,
        "MINISGL_REQUIRE_API_KEY": "1",
        "HF_TOKEN": hf_token,
        "HF_HOME": "/app/.cache/huggingface",
        "TVM_FFI_CACHE_DIR": "/app/.cache/tvm-ffi",
        "FLASHINFER_WORKSPACE_BASE": "/app/.cache/flashinfer",
    }
    env_body = "\n".join(f"{key}={value}" for key, value in remote_env.items())

    remote_script = f"""set -euo pipefail
mkdir -p ~/minisgl && cd ~/minisgl
tar -xzf ~/minisgl-lambda.tar.gz
cd deploy/lambda
cat > .env <<'ENV'
{env_body}
ENV
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker.io
  sudo usermod -aG docker ubuntu || true
fi
if ! docker compose version >/dev/null 2>&1; then
  sudo apt-get install -y -qq docker-compose-plugin || true
fi
sudo docker compose -f compose.yml up -d --build --force-recreate
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:{port}/health >/dev/null; then
    echo "Mini-SGLang is healthy"
    exit 0
  fi
  sleep 10
done
echo "Timed out waiting for Mini-SGLang health endpoint" >&2
sudo docker logs lambda-minisgl-1 2>&1 | tail -40
exit 1
"""

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
    run(ssh_cmd, input=remote_script.encode())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy Mini-SGLang to Lambda Cloud.")
    parser.add_argument("--smoke", action="store_true", help="Deploy smoke model first.")
    parser.add_argument("--reuse", action="store_true", help="Reuse first active instance.")
    parser.add_argument("--region", default=os.getenv("LAMBDA_REGION", "us-east-1"))
    args = parser.parse_args(argv)

    load_env_file(ENV_FILE)
    token = require("LAMBDA_CLOUD_API_KEY")
    api_key = require("MINISGL_API_KEY")
    hf_token = require("HF_TOKEN")
    ssh_key = Path(os.getenv("SSH_PRIVATE_KEY_PATH", str(Path.home() / ".ssh/id_ed25519")))
    instance_types = resolve_instance_types()
    model = os.getenv("MINISGL_SMOKE_MODEL", "Qwen/Qwen3-0.6B") if args.smoke else os.getenv(
        "MINISGL_MODEL", "Qwen/Qwen3-8B"
    )
    smoke_model = os.getenv("MINISGL_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    port = os.getenv("MINISGL_PORT", "1919")

    ssh_key_name = find_ssh_key_name(token, read_public_key(ssh_key))

    if args.reuse:
        reusable = list_reusable_instances(token)
        if not reusable:
            raise SystemExit("No Lambda instance to reuse.")
        instance_id = reusable[0]["id"]
        print(f"Reusing instance {instance_id} ({reusable[0].get('status')})")
    else:
        instance_id = launch_instance(token, ssh_key_name, instance_types, args.region)

    ip = wait_for_ip(token, instance_id)
    print(f"Instance IP: {ip}")
    update_env_ip(ip, ENV_FILE)
    wait_for_ssh(ip, ssh_key)

    archive = package()
    remote_deploy(
        ip,
        ssh_key,
        archive,
        model=model,
        smoke_model=smoke_model,
        port=port,
        api_key=api_key,
        hf_token=hf_token,
    )

    print("\nDeployment complete.")
    print(f"  IP: {ip}")
    print(f"  Model: {model}")
    print("  Tunnel: scripts/lambda_tunnel.sh")
    print(
        "  Infer: MINISGL_BASE_URL=http://127.0.0.1:1919/v1 "
        "MINISGL_API_KEY=<key> python scripts/minisgl_remote_infer.py 'Hello'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

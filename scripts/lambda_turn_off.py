#!/usr/bin/env python3
"""Turn off Lambda GPU instance and stop local tunnel/watchdog processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from lambda_common import (
    ENV_FILE,
    find_instance_for_ip,
    list_reusable_instances,
    load_env_file,
    request_admin_shutdown,
    require,
    stop_pid,
    terminate_instance,
    upstream_root,
)

ROOT = Path(__file__).resolve().parents[1]


def remote_docker_down(ip: str, ssh_key: Path, port: str, api_key: str) -> None:
    remote_script = f"""set -euo pipefail
if [[ -d ~/minisgl/deploy/lambda ]]; then
  cd ~/minisgl/deploy/lambda
  if [[ -f .env ]]; then set -a; source .env; set +a; fi
  curl -sf -X POST http://127.0.0.1:{port}/admin/shutdown \\
    -H 'Content-Type: application/json' \\
    -H "Authorization: Bearer ${{MINISGL_API_KEY:-{api_key}}}" || true
  sudo docker compose -f compose.yml down || true
fi
"""
    subprocess.run(
        [
            "ssh",
            "-i",
            str(ssh_key),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            f"ubuntu@{ip}",
            "bash",
            "-s",
        ],
        input=remote_script.encode(),
        check=False,
    )


def shutdown(*, terminate: bool = True, skip_remote: bool = False) -> dict:
    load_env_file(ENV_FILE)
    token = require("LAMBDA_CLOUD_API_KEY")
    ip = os.getenv("LAMBDA_PUBLIC_IP", "")
    port = os.getenv("MINISGL_PORT", "1919")
    ssh_key = Path(os.getenv("SSH_PRIVATE_KEY_PATH", str(Path.home() / ".ssh/id_ed25519")))

    # Stop local helpers first so nothing reconnects while we tear down.
    stopped = {
        "tunnel": stop_pid("tunnel.pid"),
        "watchdog": stop_pid("watchdog.pid"),
        "chat": stop_pid("chat.pid"),
    }

    request_admin_shutdown(upstream_root(), os.getenv("MINISGL_API_KEY"))

    instance = find_instance_for_ip(token, ip) or (
        list_reusable_instances(token)[0] if list_reusable_instances(token) else None
    )

    if instance and not skip_remote and ip:
        remote_docker_down(ip, ssh_key, port, require("MINISGL_API_KEY"))

    terminated = False
    if terminate and instance:
        terminate_instance(token, instance["id"])
        terminated = True

    return {
        "stopped_local": stopped,
        "instance_id": instance.get("id") if instance else None,
        "terminated": terminated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gracefully turn off Lambda Mini-SGLang.")
    parser.add_argument(
        "--keep-instance",
        action="store_true",
        help="Stop local processes and remote server but do not terminate the Lambda VM.",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Skip SSH/docker shutdown (useful when the VM is already unreachable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = shutdown(terminate=not args.keep_instance, skip_remote=args.skip_remote)
    print("Turn-off complete.")
    if result["terminated"]:
        print(f"  Terminated instance: {result['instance_id']}")
    elif result["instance_id"]:
        print(f"  Instance kept running: {result['instance_id']}")
    else:
        print("  No active Lambda instance found.")
    stopped = [name for name, ok in result["stopped_local"].items() if ok]
    if stopped:
        print(f"  Stopped local processes: {', '.join(stopped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

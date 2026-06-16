#!/usr/bin/env python3
"""Bootstrap Lambda inference without starting the chat UI (used by turn-on API)."""

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
    fetch_health,
    find_instance_for_ip,
    list_reusable_instances,
    load_env_file,
    read_pid,
    require,
    runtime_path,
    stop_pid,
    wait_for_health,
    write_pid,
)

DEPLOY = ROOT / "scripts" / "lambda_deploy.py"
WATCHDOG = ROOT / "scripts" / "lambda_idle_watchdog.py"


def ensure_tunnel(ip: str, ssh_key: Path, port: str) -> None:
    pid = read_pid("tunnel.pid")
    if pid is not None:
        try:
            os.kill(pid, 0)
            return
        except ProcessLookupError:
            stop_pid("tunnel.pid")

    log = runtime_path("tunnel.log")
    proc = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-i",
            str(ssh_key),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-L",
            f"{port}:127.0.0.1:{port}",
            f"ubuntu@{ip}",
        ],
        stdout=log.open("a"),
        stderr=subprocess.STDOUT,
    )
    write_pid("tunnel.pid", proc.pid)


def ensure_watchdog(idle_timeout_s: int) -> None:
    pid = read_pid("watchdog.pid")
    if pid is not None:
        try:
            os.kill(pid, 0)
            return
        except ProcessLookupError:
            stop_pid("watchdog.pid")

    log = runtime_path("watchdog.log")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(WATCHDOG),
            "--idle-timeout-s",
            str(idle_timeout_s),
        ],
        stdout=log.open("a"),
        stderr=subprocess.STDOUT,
    )
    write_pid("watchdog.pid", proc.pid)


def bootstrap(*, smoke: bool = False) -> dict:
    load_env_file(ENV_FILE)
    token = require("LAMBDA_CLOUD_API_KEY")
    port = os.getenv("MINISGL_PORT", "1919")
    ssh_key = Path(os.getenv("SSH_PRIVATE_KEY_PATH", str(Path.home() / ".ssh/id_ed25519")))
    idle_timeout_s = int(os.getenv("MINISGL_IDLE_TIMEOUT_S", "3600"))

    ip = os.getenv("LAMBDA_PUBLIC_IP", "")
    instance = find_instance_for_ip(token, ip) or (
        list_reusable_instances(token)[0] if list_reusable_instances(token) else None
    )

    if not fetch_health():
        deploy_args = [sys.executable, str(DEPLOY)]
        if smoke:
            deploy_args.append("--smoke")
        elif instance:
            deploy_args.append("--reuse")
        subprocess.run(deploy_args, check=True)
        load_env_file(ENV_FILE)

    ip = require("LAMBDA_PUBLIC_IP")
    ensure_tunnel(ip, ssh_key, port)
    time.sleep(1)
    wait_for_health(timeout_s=120, interval_s=3)
    ensure_watchdog(idle_timeout_s)

    return {"status": "ready", "ip": ip}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap Lambda inference for chat UI.")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = bootstrap(smoke=args.smoke)
    print(f"Bootstrap complete: {result['ip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared helpers for Lambda Cloud deploy, idle watchdog, and local chat proxy."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
RUNTIME_DIR = ROOT / ".lambda-runtime"
API_BASE = os.getenv("LAMBDA_CLOUD_API_BASE", "https://cloud.lambda.ai/api/v1")

DEFAULT_IDLE_TIMEOUT_S = int(os.getenv("MINISGL_IDLE_TIMEOUT_S", "3600"))
DEFAULT_PORT = os.getenv("MINISGL_PORT", "1919")

# Primary H100, then similar H100/A100 options when capacity is unavailable.
DEFAULT_INSTANCE_TYPE = "gpu_1x_h100_sxm5"
DEFAULT_INSTANCE_FALLBACKS = (
    "gpu_1x_h100_pcie",
    "gpu_1x_a100_sxm4",
    "gpu_1x_a100",
)


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def api_request(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    cmd = [
        "curl",
        "-sS",
        "-X",
        method,
        f"{API_BASE}{path}",
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Accept: application/json",
        "-w",
        "\n__HTTP_CODE__:%{http_code}",
    ]
    if payload is not None:
        cmd.extend(["-H", "Content-Type: application/json", "-d", json.dumps(payload)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"Lambda API {method} {path} transport error: {proc.stderr or proc.stdout}")

    body, _, code_line = proc.stdout.rpartition("\n__HTTP_CODE__:")
    code = int(code_line.strip() or "0")
    if code >= 400:
        raise SystemExit(f"Lambda API {method} {path} failed ({code}): {body.strip()}")
    if not body.strip():
        return {}
    return json.loads(body)


def list_instances(token: str) -> list[dict]:
    return api_request("GET", "/instances", token).get("data", [])


def list_reusable_instances(token: str) -> list[dict]:
    return [
        item
        for item in list_instances(token)
        if item.get("status") in {"active", "booting"}
    ]


def parse_instance_fallback_env(value: str | None = None) -> list[str]:
    """Return fallback instance types from env, or built-in defaults when unset."""
    if value is not None:
        raw = value
    elif "LAMBDA_INSTANCE_FALLBACK" in os.environ:
        raw = os.environ["LAMBDA_INSTANCE_FALLBACK"]
    else:
        return list(DEFAULT_INSTANCE_FALLBACKS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_instance_types(
    primary: str | None = None,
    fallbacks: list[str] | None = None,
) -> list[str]:
    """Build an ordered, de-duplicated list of instance types to try."""
    primary_type = primary or os.getenv("LAMBDA_INSTANCE_TYPE", DEFAULT_INSTANCE_TYPE)
    fallback_types = fallbacks if fallbacks is not None else parse_instance_fallback_env()
    ordered: list[str] = []
    seen: set[str] = set()
    for name in [primary_type, *fallback_types]:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def instance_regions_with_capacity(token: str, instance_type: str) -> list[str]:
    """Return region names that currently have launch capacity for an instance type."""
    types = api_request("GET", "/instance-types", token).get("data", {})
    info = types.get(instance_type, {})
    return [region["name"] for region in info.get("regions_with_capacity_available", [])]


def pick_region(token: str, instance_type: str, preferred: str | None) -> str:
    regions = instance_regions_with_capacity(token, instance_type)
    if not regions:
        raise SystemExit(f"No capacity for {instance_type}")
    if preferred and preferred in regions:
        return preferred
    return regions[0]


def launch_instance(
    token: str,
    ssh_key_name: str,
    instance_types: list[str],
    region: str,
    *,
    instance_name: str = "minisgl-deploy",
) -> str:
    """Try each instance type in order until one launches successfully."""
    last_error = ""
    for instance_type in instance_types:
        try:
            launch_region = pick_region(token, instance_type, region)
        except SystemExit as exc:
            last_error = str(exc)
            print(last_error)
            continue

        payload = {
            "region_name": launch_region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key_name],
            "quantity": 1,
            "name": instance_name,
        }
        try:
            resp = api_request("POST", "/instance-operations/launch", token, payload)
            instance_ids = resp.get("data", {}).get("instance_ids") or resp.get("instance_ids")
            if not instance_ids:
                raise SystemExit(f"Unexpected launch response: {resp}")
            print(f"Launched {instance_type} in {launch_region}: {instance_ids[0]}")
            return instance_ids[0]
        except SystemExit as exc:
            last_error = str(exc)
            print(f"Launch failed for {instance_type}: {last_error}")
    raise SystemExit(f"Could not launch any instance type. Last error: {last_error}")


def find_instance_for_ip(token: str, ip: str | None) -> dict | None:
    if not ip:
        return None
    for item in list_instances(token):
        if item.get("ip") == ip and item.get("status") in {"active", "booting"}:
            return item
    return None


def terminate_instance(token: str, instance_id: str) -> None:
    api_request(
        "POST",
        "/instance-operations/terminate",
        token,
        {"instance_ids": [instance_id]},
    )


def update_env_ip(ip: str, path: Path = ENV_FILE) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("LAMBDA_PUBLIC_IP="):
            out.append(f"LAMBDA_PUBLIC_IP={ip}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"LAMBDA_PUBLIC_IP={ip}")
    path.write_text("\n".join(out) + "\n")


def runtime_path(name: str) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR / name


def write_pid(name: str, pid: int) -> None:
    runtime_path(name).write_text(str(pid))


def read_pid(name: str) -> int | None:
    path = runtime_path(name)
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def stop_pid(name: str, *, sig: int = signal.SIGTERM) -> bool:
    pid = read_pid(name)
    if pid is None:
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        runtime_path(name).unlink(missing_ok=True)
        return False
    runtime_path(name).unlink(missing_ok=True)
    return True


def upstream_root(base_url: str | None = None) -> str:
    value = (base_url or os.getenv("MINISGL_BASE_URL", "http://127.0.0.1:1919")).rstrip("/")
    if value.endswith("/v1"):
        return value[:-3]
    return value


def fetch_status(base_url: str | None = None, *, timeout_s: float = 5.0) -> tuple[int, dict | str]:
    url = f"{upstream_root(base_url)}/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)


def fetch_health(base_url: str | None = None, *, timeout_s: float = 5.0) -> bool:
    url = f"{upstream_root(base_url)}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def wait_for_health(
    base_url: str | None = None,
    *,
    timeout_s: int = 600,
    interval_s: int = 10,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fetch_health(base_url):
            return
        time.sleep(interval_s)
    raise SystemExit("Timed out waiting for Mini-SGLang /health")


def request_admin_shutdown(base_url: str | None = None, api_key: str | None = None) -> bool:
    url = f"{upstream_root(base_url)}/admin/shutdown"
    api_key = api_key or os.getenv("MINISGL_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=b"{}", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def seconds_until_idle(status_payload: dict, idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S) -> int | None:
    """Return seconds remaining before idle shutdown, or None if unknown."""
    last = status_payload.get("last_activity_at")
    if last is None:
        return None
    timeout = int(status_payload.get("idle_timeout_seconds") or idle_timeout_s)
    elapsed = time.time() - float(last)
    return max(0, int(timeout - elapsed))


def read_watchdog_state() -> dict:
    path = runtime_path("watchdog-state.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_watchdog_state(payload: dict) -> None:
    runtime_path("watchdog-state.json").write_text(json.dumps(payload, indent=2) + "\n")


def is_idle_expired(status_payload: dict, idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S) -> bool:
    remaining = seconds_until_idle(status_payload, idle_timeout_s)
    return remaining is not None and remaining <= 0

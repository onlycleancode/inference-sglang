#!/usr/bin/env python3
"""Serve deploy/lambda/chat.html and proxy Mini-SGLang API calls (avoids browser CORS).

Usage:
  ./scripts/lambda_tunnel.sh          # separate terminal
  python scripts/serve_lambda_chat.py # then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lambda_common import (  # noqa: E402
    DEFAULT_IDLE_TIMEOUT_S,
    ENV_FILE,
    fetch_health,
    fetch_status,
    find_instance_for_ip,
    list_reusable_instances,
    load_env_file,
    read_watchdog_state,
)
CHAT_HTML = ROOT / "deploy" / "lambda" / "chat.html"
CONFIG_MARKER = "<!-- minisgl-chat-config -->"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _split_host_port(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not port:
        raise SystemExit(f"Invalid listen address {value!r}; expected host:port")
    return host, int(port)


def _upstream_root(value: str) -> str:
    trimmed = value.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[:-3]
    return trimmed


load_env_file(ENV_FILE)
DEFAULT_UPSTREAM = _upstream_root(os.getenv("MINISGL_BASE_URL", "http://127.0.0.1:1919"))
DEFAULT_LISTEN = os.getenv("MINISGL_CHAT_LISTEN", "127.0.0.1:8765")

_bootstrap_lock = threading.Lock()
_bootstrap_state: dict = {"status": "idle", "message": ""}


def collect_local_status(upstream_root: str) -> dict:
    load_env_file(ENV_FILE)
    healthy = fetch_health(upstream_root)
    code, payload = fetch_status(upstream_root)
    watchdog = read_watchdog_state()

    instance = None
    try:
        token = os.getenv("LAMBDA_CLOUD_API_KEY")
        ip = os.getenv("LAMBDA_PUBLIC_IP", "")
        if token:
            instance = find_instance_for_ip(token, ip) or (
                list_reusable_instances(token)[0] if list_reusable_instances(token) else None
            )
    except SystemExit:
        instance = None

    server_on = healthy and code == 200
    status_payload = payload if isinstance(payload, dict) else {}
    return {
        "server_on": server_on,
        "health_ok": healthy,
        "upstream_status_code": code,
        "server_status": status_payload,
        "seconds_until_idle": status_payload.get("seconds_until_idle"),
        "idle_timeout_seconds": status_payload.get("idle_timeout_seconds", DEFAULT_IDLE_TIMEOUT_S),
        "lambda_instance": {
            "id": instance.get("id") if instance else None,
            "status": instance.get("status") if instance else None,
            "ip": instance.get("ip") if instance else os.getenv("LAMBDA_PUBLIC_IP"),
        },
        "watchdog": watchdog,
        "bootstrap": dict(_bootstrap_state),
    }


def _run_bootstrap_background() -> None:
    global _bootstrap_state
    with _bootstrap_lock:
        if _bootstrap_state.get("status") == "running":
            return
        _bootstrap_state = {"status": "running", "message": "Starting Lambda GPU…"}

    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "lambda_local_bootstrap.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        message = proc.stdout.strip() or "Server is ready."
        status = "ready"
    except subprocess.CalledProcessError as exc:
        status = "error"
        message = (exc.stderr or exc.stdout or str(exc)).strip()

    with _bootstrap_lock:
        _bootstrap_state = {"status": status, "message": message}


def start_bootstrap_async() -> dict:
    thread = threading.Thread(target=_run_bootstrap_background, daemon=True)
    thread.start()
    return {"status": "starting", "message": "Bootstrap started."}


def run_shutdown() -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "lambda_turn_off.py")],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "shutdown failed")
    return {"status": "off", "message": proc.stdout.strip()}


def _browser_base_url(listen: str) -> str:
    """URL the chat page should fetch — always the local proxy, never upstream."""
    host, port = _split_host_port(listen)
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}/v1"


def render_chat_html(listen: str = DEFAULT_LISTEN) -> bytes:
    template = CHAT_HTML.read_text(encoding="utf-8")
    config: dict[str, str] = {"baseUrl": _browser_base_url(listen)}
    api_key = os.getenv("MINISGL_API_KEY", "").strip()
    if api_key:
        config["apiKey"] = api_key
    model = os.getenv("MINISGL_MODEL", "").strip()
    if model:
        config["model"] = model

    if config:
        script = (
            '<script id="minisgl-config">'
            f"window.__MINISGL_CHAT_CONFIG__ = {json.dumps(config)};"
            "</script>"
        )
        html = template.replace(CONFIG_MARKER, script)
    else:
        html = template.replace(CONFIG_MARKER, "")

    return html.encode("utf-8")


class ChatProxyHandler(BaseHTTPRequestHandler):
    upstream_root: str = DEFAULT_UPSTREAM
    listen_addr: str = DEFAULT_LISTEN

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/chat.html"):
            body = render_chat_html(self.listen_addr)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/local/status":
            self._send_json(200, collect_local_status(self.upstream_root))
            return

        if self.path.startswith(("/v1", "/health", "/status")):
            self._proxy("GET")
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/local/turn-on":
            self._send_json(200, start_bootstrap_async())
            return

        if self.path == "/api/local/shutdown":
            try:
                result = run_shutdown()
                self._send_json(200, result)
            except RuntimeError as exc:
                self._send_json(500, {"status": "error", "message": str(exc)})
            return

        if self.path.startswith("/v1") or self.path == "/admin/shutdown":
            self._proxy("POST")
            return

        self.send_error(404, "Not found")

    def _proxy(self, method: str) -> None:
        upstream = f"{self.upstream_root}{self.path}"
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None

        headers = {}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type

        req = urllib.request.Request(upstream, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = resp.read()
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() in {"transfer-encoding", "connection"}:
                        continue
                    self.send_header(key, value)
                self._send_cors()
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(payload)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Lambda chat test UI with API proxy.")
    parser.add_argument(
        "--listen",
        default=DEFAULT_LISTEN,
        help="Host:port to bind (env: MINISGL_CHAT_LISTEN).",
    )
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help="Mini-SGLang root URL without /v1 suffix (env: MINISGL_BASE_URL).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file(ENV_FILE)

    args = build_parser().parse_args(argv)
    if not CHAT_HTML.is_file():
        print(f"Missing chat page: {CHAT_HTML}", file=sys.stderr)
        return 1

    host, port = _split_host_port(args.listen)
    upstream = args.upstream.rstrip("/")
    if upstream.endswith("/v1"):
        upstream = upstream[:-3]
    ChatProxyHandler.upstream_root = upstream
    ChatProxyHandler.listen_addr = args.listen

    server = ThreadingHTTPServer((host, port), ChatProxyHandler)
    print(f"Serving chat UI at http://{host}:{port}/")
    print(f"Proxying /v1/*, /health, /status -> {upstream}")
    print("Local control: GET /api/local/status, POST /api/local/turn-on, POST /api/local/shutdown")
    if os.getenv("MINISGL_API_KEY"):
        print("Loaded MINISGL_API_KEY from .env (injected into chat UI).")
    else:
        print("MINISGL_API_KEY not set; enter the key manually in the chat UI.")
    print("Leave lambda_tunnel.sh running in another terminal.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

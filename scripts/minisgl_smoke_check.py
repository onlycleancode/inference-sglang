#!/usr/bin/env python3
"""Smoke-check a remote Mini-SGLang deployment (models, chat, auth)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check a Mini-SGLang endpoint.")
    parser.add_argument(
        "--base-url",
        default=_env("MINISGL_BASE_URL", "http://127.0.0.1:1919/v1"),
        help="OpenAI-compatible base URL without trailing slash issues (env: MINISGL_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=_env("MINISGL_API_KEY"),
        help="Bearer token (env: MINISGL_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=_env("MINISGL_MODEL", "Qwen/Qwen3-8B"),
        help="Model id for chat completion checks (env: MINISGL_MODEL).",
    )
    parser.add_argument(
        "--skip-stream",
        action="store_true",
        help="Skip streaming completion check.",
    )
    return parser


def _request(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict | None = None,
    stream: bool = False,
) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if stream:
                chunks = []
                for line in resp:
                    chunks.append(line.decode())
                return resp.status, "".join(chunks)
            body = resp.read().decode()
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        return exc.code, body


def _root_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[:-3] or base_url
    return base_url.rstrip("/")


def check_health(base_url: str) -> None:
    url = f"{_root_url(base_url)}/health"
    status, body = _request("GET", url)
    assert status == 200, f"health failed: {status} {body}"
    print("OK  GET /health")


def check_models(base_url: str, api_key: str) -> None:
    url = f"{base_url.rstrip('/')}/models"
    status, body = _request("GET", url, api_key=api_key)
    assert status == 200, f"models failed: {status} {body}"
    print("OK  GET /v1/models")


def check_chat(base_url: str, api_key: str, model: str, *, stream: bool) -> None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 16,
        "stream": stream,
    }
    status, body = _request("POST", url, api_key=api_key, payload=payload, stream=stream)
    assert status == 200, f"chat failed: {status} {body}"
    label = "streaming" if stream else "non-streaming"
    print(f"OK  POST /v1/chat/completions ({label})")


def check_auth_rejects(base_url: str) -> None:
    url = f"{base_url.rstrip('/')}/models"
    status, _ = _request("GET", url, api_key=None)
    assert status == 401, f"expected 401 without token, got {status}"

    status, _ = _request("GET", url, api_key="definitely-wrong")
    assert status == 401, f"expected 401 with wrong token, got {status}"
    print("OK  auth rejects missing/wrong token")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.api_key:
        print("MINISGL_API_KEY is required.", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    print(f"Smoke-checking {base_url}")

    check_health(base_url)
    check_models(base_url, args.api_key)
    check_chat(base_url, args.api_key, args.model, stream=False)
    if not args.skip_stream:
        check_chat(base_url, args.api_key, args.model, stream=True)
    check_auth_rejects(base_url)

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

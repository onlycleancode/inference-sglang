#!/usr/bin/env python3
"""Idle watchdog: terminate Lambda GPU when inference is idle too long."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from lambda_common import (
    DEFAULT_IDLE_TIMEOUT_S,
    ENV_FILE,
    fetch_status,
    is_idle_expired,
    load_env_file,
    read_watchdog_state,
    runtime_path,
    upstream_root,
    write_watchdog_state,
)

ROOT = Path(__file__).resolve().parents[1]


def run_watchdog(
    *,
    idle_timeout_s: int,
    poll_interval_s: int,
    base_url: str,
    once: bool = False,
) -> int:
    load_env_file(ENV_FILE)

    # Import here so tests can patch shutdown without circular imports.
    from lambda_turn_off import shutdown

    while True:
        code, payload = fetch_status(base_url)
        now = time.time()
        state = {
            "checked_at": now,
            "upstream_status_code": code,
            "idle_timeout_seconds": idle_timeout_s,
        }

        if code == 200 and isinstance(payload, dict):
            state["server_status"] = payload
            remaining = payload.get("seconds_until_idle")
            if remaining is None and payload.get("last_activity_at") is not None:
                timeout = int(payload.get("idle_timeout_seconds") or idle_timeout_s)
                elapsed = now - float(payload["last_activity_at"])
                remaining = max(0, int(timeout - elapsed))
            state["seconds_until_idle"] = remaining

            if is_idle_expired(payload, idle_timeout_s):
                state["action"] = "idle_shutdown"
                write_watchdog_state(state)
                print(
                    f"No inference activity for {idle_timeout_s}s — shutting down Lambda.",
                    file=sys.stderr,
                )
                shutdown(terminate=True, skip_remote=False)
                return 0

            state["action"] = "monitoring"
        else:
            state["server_status"] = payload
            state["action"] = "upstream_unreachable"

        write_watchdog_state(state)

        if once:
            return 0

        time.sleep(poll_interval_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch Mini-SGLang activity and terminate Lambda on idle timeout.",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=int,
        default=int(os.getenv("MINISGL_IDLE_TIMEOUT_S", str(DEFAULT_IDLE_TIMEOUT_S))),
        help="Seconds without inference before shutdown (default: 3600).",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=int,
        default=int(os.getenv("MINISGL_IDLE_POLL_S", "60")),
        help="Seconds between /status checks (default: 60).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("MINISGL_BASE_URL", "http://127.0.0.1:1919/v1"),
        help="Mini-SGLang base URL (env: MINISGL_BASE_URL).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single idle check (for tests and manual verification).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_watchdog(
        idle_timeout_s=args.idle_timeout_s,
        poll_interval_s=args.poll_interval_s,
        base_url=upstream_root(args.base_url),
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())

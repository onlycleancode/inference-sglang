from __future__ import annotations

import os
import time

# Updated on startup and on each inference request.
_last_activity_at: float = time.time()


def _idle_timeout_seconds() -> int:
    return int(os.getenv("MINISGL_IDLE_TIMEOUT_S", "3600"))


def touch_activity() -> None:
    global _last_activity_at
    _last_activity_at = time.time()


def activity_snapshot() -> dict:
    now = time.time()
    elapsed = now - _last_activity_at
    idle_timeout = _idle_timeout_seconds()
    remaining = max(0, int(idle_timeout - elapsed))
    return {
        "status": "ok",
        "last_activity_at": _last_activity_at,
        "idle_timeout_seconds": idle_timeout,
        "seconds_since_activity": int(elapsed),
        "seconds_until_idle": remaining,
    }

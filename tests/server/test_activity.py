from __future__ import annotations

import time

import pytest

from minisgl.server.activity import activity_snapshot, touch_activity


def test_activity_snapshot_includes_idle_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINISGL_IDLE_TIMEOUT_S", "120")
    fixed = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: fixed)

    touch_activity()
    snapshot = activity_snapshot()

    assert snapshot["status"] == "ok"
    assert snapshot["idle_timeout_seconds"] == 120
    assert snapshot["seconds_since_activity"] == 0
    assert snapshot["seconds_until_idle"] == 120


def test_touch_activity_updates_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([100.0, 250.0])
    monkeypatch.setattr(time, "time", lambda: next(times))
    monkeypatch.setenv("MINISGL_IDLE_TIMEOUT_S", "60")

    touch_activity()
    later = activity_snapshot()

    assert later["seconds_since_activity"] == 150
    assert later["seconds_until_idle"] == 0

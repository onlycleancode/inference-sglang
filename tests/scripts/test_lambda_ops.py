from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_idle_helpers() -> None:
    common = _load_module("lambda_common_test", SCRIPTS / "lambda_common.py")
    payload = {
        "last_activity_at": 1000.0,
        "idle_timeout_seconds": 3600,
    }

    remaining = common.seconds_until_idle(payload, idle_timeout_s=3600)
    assert remaining is not None
    assert remaining <= 3600

    expired = common.is_idle_expired(
        {"last_activity_at": 1.0, "idle_timeout_seconds": 10},
        idle_timeout_s=10,
    )
    assert expired is True


def test_resolve_instance_types_uses_defaults(monkeypatch) -> None:
    common = _load_module("lambda_common_resolve_defaults", SCRIPTS / "lambda_common.py")
    monkeypatch.delenv("LAMBDA_INSTANCE_TYPE", raising=False)
    monkeypatch.delenv("LAMBDA_INSTANCE_FALLBACK", raising=False)

    assert common.resolve_instance_types() == [
        "gpu_1x_h100_sxm5",
        "gpu_1x_h100_pcie",
        "gpu_1x_a100_sxm4",
        "gpu_1x_a100",
    ]


def test_resolve_instance_types_respects_env_overrides(monkeypatch) -> None:
    common = _load_module("lambda_common_resolve_env", SCRIPTS / "lambda_common.py")
    monkeypatch.setenv("LAMBDA_INSTANCE_TYPE", "gpu_1x_h100_pcie")
    monkeypatch.setenv("LAMBDA_INSTANCE_FALLBACK", "gpu_1x_a100,gpu_1x_a100_sxm4")

    assert common.resolve_instance_types() == [
        "gpu_1x_h100_pcie",
        "gpu_1x_a100",
        "gpu_1x_a100_sxm4",
    ]


def test_resolve_instance_types_empty_fallback_env_disables_defaults(monkeypatch) -> None:
    common = _load_module("lambda_common_resolve_empty", SCRIPTS / "lambda_common.py")
    monkeypatch.delenv("LAMBDA_INSTANCE_TYPE", raising=False)
    monkeypatch.setenv("LAMBDA_INSTANCE_FALLBACK", "")

    assert common.resolve_instance_types() == ["gpu_1x_h100_sxm5"]


def test_resolve_instance_types_deduplicates_primary(monkeypatch) -> None:
    common = _load_module("lambda_common_resolve_dedupe", SCRIPTS / "lambda_common.py")
    monkeypatch.setenv("LAMBDA_INSTANCE_TYPE", "gpu_1x_a100")
    monkeypatch.setenv("LAMBDA_INSTANCE_FALLBACK", "gpu_1x_a100,gpu_1x_h100_sxm5")

    assert common.resolve_instance_types() == [
        "gpu_1x_a100",
        "gpu_1x_h100_sxm5",
    ]


def test_instance_regions_with_capacity(monkeypatch) -> None:
    common = _load_module("lambda_common_capacity", SCRIPTS / "lambda_common.py")

    def fake_api_request(method: str, path: str, token: str, payload=None):
        assert method == "GET"
        assert path == "/instance-types"
        return {
            "data": {
                "gpu_1x_h100_sxm5": {
                    "regions_with_capacity_available": [{"name": "us-west-2"}],
                },
                "gpu_1x_a100": {
                    "regions_with_capacity_available": [],
                },
            }
        }

    monkeypatch.setattr(common, "api_request", fake_api_request)

    assert common.instance_regions_with_capacity("token", "gpu_1x_h100_sxm5") == ["us-west-2"]
    assert common.instance_regions_with_capacity("token", "gpu_1x_a100") == []


def test_pick_region_prefers_requested_region(monkeypatch) -> None:
    common = _load_module("lambda_common_pick_region", SCRIPTS / "lambda_common.py")
    monkeypatch.setattr(
        common,
        "instance_regions_with_capacity",
        lambda *_a, **_k: ["us-west-2", "us-east-1"],
    )

    assert common.pick_region("token", "gpu_1x_a100", "us-east-1") == "us-east-1"


def test_pick_region_raises_when_no_capacity(monkeypatch) -> None:
    common = _load_module("lambda_common_no_capacity", SCRIPTS / "lambda_common.py")
    monkeypatch.setattr(common, "instance_regions_with_capacity", lambda *_a, **_k: [])

    with pytest.raises(SystemExit, match="No capacity for gpu_1x_h100_sxm5"):
        common.pick_region("token", "gpu_1x_h100_sxm5", "us-east-1")


def test_launch_instance_falls_back_when_primary_unavailable(monkeypatch, capsys) -> None:
    common = _load_module("lambda_common_launch_fallback", SCRIPTS / "lambda_common.py")
    calls: list[tuple[str, str, dict | None]] = []

    def fake_api_request(method: str, path: str, token: str, payload=None):
        calls.append((method, path, payload))
        if method == "GET" and path == "/instance-types":
            return {
                "data": {
                    "gpu_1x_h100_sxm5": {"regions_with_capacity_available": []},
                    "gpu_1x_a100": {"regions_with_capacity_available": [{"name": "us-east-1"}]},
                }
            }
        if method == "POST" and path == "/instance-operations/launch":
            assert payload is not None
            return {"data": {"instance_ids": ["inst-fallback"]}}
        raise AssertionError(f"Unexpected API call: {method} {path}")

    monkeypatch.setattr(common, "api_request", fake_api_request)

    instance_id = common.launch_instance(
        "token",
        "my-key",
        ["gpu_1x_h100_sxm5", "gpu_1x_a100"],
        "us-east-1",
    )

    assert instance_id == "inst-fallback"
    launch_calls = [call for call in calls if call[0] == "POST"]
    assert len(launch_calls) == 1
    assert launch_calls[0][2]["instance_type_name"] == "gpu_1x_a100"
    assert "No capacity for gpu_1x_h100_sxm5" in capsys.readouterr().out


def test_launch_instance_fails_when_all_types_unavailable(monkeypatch) -> None:
    common = _load_module("lambda_common_launch_fail", SCRIPTS / "lambda_common.py")
    monkeypatch.setattr(
        common,
        "instance_regions_with_capacity",
        lambda *_a, **_k: [],
    )

    with pytest.raises(SystemExit, match="Could not launch any instance type"):
        common.launch_instance(
            "token",
            "my-key",
            ["gpu_1x_h100_sxm5", "gpu_1x_a100"],
            "us-east-1",
        )


def test_collect_local_status_offline(monkeypatch) -> None:
    serve = _load_module("serve_lambda_chat_test", SCRIPTS / "serve_lambda_chat.py")

    monkeypatch.setattr(serve, "fetch_health", lambda *_a, **_k: False)
    monkeypatch.setattr(serve, "fetch_status", lambda *_a, **_k: (0, "connection refused"))
    monkeypatch.setattr(serve, "read_watchdog_state", lambda: {})
    monkeypatch.setattr(serve, "find_instance_for_ip", lambda *_a, **_k: None)
    monkeypatch.setattr(serve, "list_reusable_instances", lambda *_a, **_k: [])

    status = serve.collect_local_status("http://127.0.0.1:1919")
    assert status["server_on"] is False
    assert status["health_ok"] is False

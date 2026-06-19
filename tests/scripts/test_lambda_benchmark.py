"""Tests for multi-model Lambda benchmark harness."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
BENCHMARK = SCRIPTS / "benchmark"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark(name: str, filename: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return _load_module(name, BENCHMARK / filename)


def test_select_instance_type_prefers_h100(monkeypatch) -> None:
    inst = _load_benchmark("bench_instance_h100", "instance.py")
    calls: list[str] = []

    def fake_regions(_token: str, instance_type: str) -> list[str]:
        calls.append(instance_type)
        if instance_type == "gpu_1x_h100_sxm5":
            return ["us-east-1"]
        return []

    monkeypatch.setattr(inst, "instance_regions_with_capacity", fake_regions)
    monkeypatch.setattr(inst, "pick_region", lambda *_a, **_k: "us-east-1")

    selected = inst.select_instance_type_with_capacity(
        "token",
        instance_types=["gpu_1x_h100_sxm5", "gpu_1x_h100_pcie", "gpu_1x_a100"],
    )
    assert selected.instance_type == "gpu_1x_h100_sxm5"
    assert calls[0] == "gpu_1x_h100_sxm5"


def test_select_instance_type_falls_back(monkeypatch) -> None:
    inst = _load_benchmark("bench_instance_fallback", "instance.py")

    def fake_regions(_token: str, instance_type: str) -> list[str]:
        if instance_type == "gpu_1x_a100":
            return ["us-west-2"]
        return []

    monkeypatch.setattr(inst, "instance_regions_with_capacity", fake_regions)
    monkeypatch.setattr(inst, "pick_region", lambda *_a, **_k: "us-west-2")

    selected = inst.select_instance_type_with_capacity(
        "token",
        instance_types=["gpu_1x_h100_sxm5", "gpu_1x_a100"],
    )
    assert selected.instance_type == "gpu_1x_a100"


def test_select_instance_type_no_capacity_raises(monkeypatch) -> None:
    inst = _load_benchmark("bench_instance_none", "instance.py")
    monkeypatch.setattr(inst, "instance_regions_with_capacity", lambda *_a, **_k: [])

    with pytest.raises(RuntimeError, match="No Lambda capacity"):
        inst.select_instance_type_with_capacity("token", instance_types=["gpu_1x_h100_sxm5"])


def test_cleanup_partial_launch_terminates_tracked_ids(monkeypatch) -> None:
    cleanup_mod = _load_benchmark("bench_cleanup_partial", "cleanup.py")
    terminated: list[str] = []

    def fake_terminate(_token: str, instance_id: str) -> None:
        terminated.append(instance_id)

    def fake_list(_token: str) -> list[dict]:
        return []

    mgr = cleanup_mod.BenchmarkCleanupManager(
        token="token",
        run_id="run-1",
        terminate_fn=fake_terminate,
        list_instances_fn=fake_list,
    )
    mgr.register_instances(["inst-1", "inst-2"])
    mgr.cleanup(reason="partial_launch")
    assert terminated == ["inst-1", "inst-2"]


def test_cleanup_deploy_failure_terminates(monkeypatch) -> None:
    cleanup_mod = _load_benchmark("bench_cleanup_deploy", "cleanup.py")
    terminated: list[str] = []

    mgr = cleanup_mod.BenchmarkCleanupManager(
        token="token",
        run_id="run-2",
        terminate_fn=lambda _t, i: terminated.append(i),
        list_instances_fn=lambda _t: [],
    )
    mgr.register_instances(["inst-a"])
    try:
        with mgr:
            raise RuntimeError("deploy failed")
    except RuntimeError:
        pass
    assert terminated == ["inst-a"]


def test_cleanup_benchmark_failure_terminates() -> None:
    cleanup_mod = _load_benchmark("bench_cleanup_bench", "cleanup.py")
    terminated: list[str] = []

    mgr = cleanup_mod.BenchmarkCleanupManager(
        token="token",
        run_id="run-3",
        terminate_fn=lambda _t, i: terminated.append(i),
        list_instances_fn=lambda _t: [],
    )
    mgr.register_instances(["inst-x"])
    try:
        with mgr:
            raise ValueError("benchmark failed")
    except ValueError:
        pass
    assert terminated == ["inst-x"]


def test_cleanup_keyboard_interrupt_terminates(monkeypatch) -> None:
    cleanup_mod = _load_benchmark("bench_cleanup_sig", "cleanup.py")
    terminated: list[str] = []

    mgr = cleanup_mod.BenchmarkCleanupManager(
        token="token",
        run_id="run-4",
        terminate_fn=lambda _t, i: terminated.append(i),
        list_instances_fn=lambda _t: [],
    )
    mgr.register_instances(["inst-z"])
    mgr.install_signal_handlers()
    with pytest.raises(SystemExit):
        mgr._handle_signal(signal.SIGINT, None)
    assert terminated == ["inst-z"]


def test_duckdb_schema_and_inserts(tmp_path: Path) -> None:
    store_mod = _load_benchmark("bench_store", "duckdb_store.py")
    db_path = tmp_path / "bench.duckdb"
    store = store_mod.BenchmarkStore(db_path)
    store.insert_run(
        run_id="r1",
        dataset_path="/tmp/data.jsonl",
        dataset_hash="abc",
        concurrency_levels=[1, 4],
        instance_type="gpu_1x_h100_sxm5",
        region="us-east-1",
        git_sha="deadbeef",
        archive_path="/tmp/archive.tar.gz",
        hourly_usd=3.29,
        node_count=3,
    )
    store.insert_node(
        run_id="r1",
        node_index=0,
        model_id="Qwen/Qwen3-8B",
        instance_id="i1",
        ip="1.2.3.4",
        local_port=19191,
        model_load_s=120.0,
        warmup_s=5.0,
    )
    started = datetime.now(timezone.utc)
    store.insert_request(
        request_id="req-1",
        run_id="r1",
        node_index=0,
        model_id="Qwen/Qwen3-8B",
        concurrency=1,
        dataset_row_id="row-0",
        dataset_row_index=0,
        status="ok",
        error_message=None,
        ttft_ms=50.0,
        tpot_ms=10.0,
        e2e_s=1.2,
        output_tokens=64,
        prompt_len_bucket="128-512",
        output_len_bucket="32-128",
        output_tokens_per_sec=53.3,
        started_at=started,
        finished_at=started,
    )
    store.insert_stream_tokens("req-1", [0.0, 0.05, 0.06, 0.07])
    assert store.get_next_row_index("r1") == 0
    store.set_next_row_index("r1", 12)
    assert store.get_next_row_index("r1") == 12
    store.insert_cleanup(
        run_id="r1",
        instance_ids=["i1"],
        success=True,
        remaining_instances=[],
        message="ok",
    )
    store.close()

    conn = duckdb.connect(str(db_path), read_only=True)
    assert conn.execute("SELECT count(*) FROM benchmark_runs").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM benchmark_requests").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM benchmark_stream_tokens").fetchone()[0] == 4
    assert conn.execute("SELECT next_row_index FROM benchmark_control WHERE run_id = 'r1'").fetchone()[0] == 12
    conn.close()


def test_default_dataset_is_alpaca_2k() -> None:
    cfg = _load_benchmark("bench_config_alpaca", "config.py")
    ds_mod = _load_benchmark("bench_dataset_alpaca", "dataset.py")

    assert cfg.BenchmarkConfig().dataset_path == cfg.ALPACA_2K_DATASET
    dataset = ds_mod.load_dataset(cfg.ALPACA_2K_DATASET)
    assert dataset.size == 2000
    assert dataset.rows[0].row_id == "alpaca-2k-0000"
    assert dataset.rows[0].metadata["source"] == "mhenrichsen/alpaca_2k_test"


def test_dataset_parsing_prompt_messages_and_hash(tmp_path: Path) -> None:
    ds_mod = _load_benchmark("bench_dataset", "dataset.py")
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"a","prompt":"hello","max_tokens":16}',
                '{"id":"b","messages":[{"role":"user","content":"hi"}]}',
                '{"prompt":"default tokens only"}',
            ]
        )
        + "\n"
    )
    dataset = ds_mod.load_dataset(path)
    assert dataset.size == 3
    assert dataset.rows[0].max_tokens == 16
    assert dataset.rows[1].messages is not None
    assert dataset.rows[2].max_tokens is None
    assert len(dataset.content_hash) == 64

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        ds_mod.load_dataset(bad)

    missing = tmp_path / "missing.jsonl"
    missing.write_text('{"id":"x"}\n')
    with pytest.raises(ValueError, match="require 'prompt' or 'messages'"):
        ds_mod.load_dataset(missing)


def test_dataset_hash_is_stable(tmp_path: Path) -> None:
    ds_mod = _load_benchmark("bench_dataset_hash", "dataset.py")
    path = tmp_path / "data.jsonl"
    content = '{"prompt":"stable"}\n{"prompt":"rows"}\n'
    path.write_text(content)
    h1 = ds_mod.load_dataset(path).content_hash
    h2 = ds_mod.load_dataset(path).content_hash
    assert h1 == h2


def test_metric_calculations_from_timestamps() -> None:
    metrics_mod = _load_benchmark("bench_metrics", "metrics.py")
    tics = [0.0, 0.1, 0.15, 0.20, 0.28]
    m = metrics_mod.compute_request_metrics(tics)
    assert m.ttft_s == pytest.approx(0.1)
    assert m.e2e_s == pytest.approx(0.28)
    assert m.output_tokens == 4
    assert m.tpot_s == pytest.approx((0.05 + 0.05 + 0.08) / 3)
    summary = metrics_mod.summarize_latencies([0.1, 0.2, 0.3])
    assert summary["p50"] == pytest.approx(0.2)


def test_dataset_request_body_uses_top_level_sampling_fields() -> None:
    ds_mod = _load_benchmark("bench_dataset_body", "dataset.py")
    row = ds_mod.DatasetRow(
        row_index=0,
        row_id="r0",
        prompt="hello",
        messages=None,
        max_tokens=32,
    )
    body = row.request_body(default_max_tokens=128)
    assert body["ignore_eos"] is True
    assert body["top_k"] == 1
    assert "extra_body" not in body

    warmup = ds_mod.warmup_request_body("warmup prompt")
    assert warmup["ignore_eos"] is True
    assert warmup["top_k"] == 1
    assert "extra_body" not in warmup


def test_node_upsert_preserves_model_load_s_on_warmup_update(tmp_path: Path) -> None:
    store_mod = _load_benchmark("bench_store_upsert", "duckdb_store.py")
    db_path = tmp_path / "upsert.duckdb"
    store = store_mod.BenchmarkStore(db_path)
    store.insert_node(
        run_id="r1",
        node_index=0,
        model_id="Qwen/Qwen3-8B",
        instance_id="i1",
        ip="1.2.3.4",
        local_port=19191,
        model_load_s=120.0,
    )
    store.insert_node(
        run_id="r1",
        node_index=0,
        model_id="Qwen/Qwen3-8B",
        instance_id="i1",
        ip="1.2.3.4",
        local_port=19191,
        warmup_s=8.5,
    )
    store.close()

    conn = duckdb.connect(str(db_path), read_only=True)
    row = conn.execute(
        "SELECT model_load_s, warmup_s FROM benchmark_nodes WHERE run_id = 'r1'"
    ).fetchone()
    conn.close()
    assert row == (120.0, 8.5)


def test_request_throughput_uses_wall_clock() -> None:
    metrics_mod = _load_benchmark("bench_metrics_tp", "metrics.py")
    tp = metrics_mod.request_throughput_per_s(
        request_count=10,
        wall_start_s=0.0,
        wall_end_s=5.0,
    )
    assert tp == pytest.approx(2.0)


def test_dashboard_throughput_uses_timedelta_wall_clock() -> None:
    dash = _load_benchmark("bench_dashboard_tp", "dashboard.py")
    import pandas as pd

    t0 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    ok = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "concurrency": [1, 1],
            "request_id": ["r1", "r2"],
            "started_at": [t0, t0 + pd.Timedelta(seconds=1)],
            "finished_at": [t0 + pd.Timedelta(seconds=2), t0 + pd.Timedelta(seconds=3)],
        }
    )
    tp = dash._throughput_by_group(ok)
    assert len(tp) == 1
    assert tp.iloc[0]["req_per_s"] == pytest.approx(2.0 / 3.0)
    assert tp.iloc[0]["req_per_s"] < 10.0


def test_stream_completion_skips_finish_reason_stop_chunk() -> None:
    runner_mod = _load_benchmark("bench_runner_stream", "runner.py")
    metrics_mod = _load_benchmark("bench_metrics_stream", "metrics.py")

    assert runner_mod._stream_chunk_carries_content({"choices": [{"delta": {"content": "a"}}]})
    assert not runner_mod._stream_chunk_carries_content(
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    )

    tics = [0.0]
    chunks = [
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    for chunk in chunks:
        if runner_mod._stream_chunk_carries_content(chunk):
            tics.append(tics[-1] + 0.01)

    metrics = metrics_mod.compute_request_metrics(tics)
    assert metrics.output_tokens == 2


def test_dashboard_filter_keeps_null_output_bucket_errors() -> None:
    dash = _load_benchmark("bench_dashboard_filter", "dashboard.py")
    import pandas as pd

    requests = pd.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "concurrency": [1, 1],
            "prompt_len_bucket": ["128-512", "128-512"],
            "output_len_bucket": ["32-128", None],
            "status": ["ok", "error"],
        }
    )
    filtered = dash._filter_requests(
        requests,
        models=["m1"],
        concurrency=[1],
        prompt_buckets=["128-512"],
        output_buckets=["32-128"],
        statuses=["ok", "error"],
    )
    assert len(filtered) == 2


def test_dashboard_control_selects_next_and_random_rows(tmp_path: Path) -> None:
    ds_mod = _load_benchmark("bench_dataset_control", "dataset.py")
    control_mod = _load_benchmark("bench_control_select", "control.py")

    path = tmp_path / "data.jsonl"
    rows = [{"id": str(i), "prompt": f"p{i}", "max_tokens": 8} for i in range(5)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dataset = ds_mod.load_dataset(path)

    selected, cursor = control_mod.select_next_rows(dataset, 1, 3)
    assert [row.row_id for row in selected] == ["1", "2", "3"]
    assert cursor == 4

    selected, cursor = control_mod.select_next_rows(dataset, 4, 10)
    assert [row.row_id for row in selected] == ["4"]
    assert cursor == 5

    random_rows = control_mod.select_random_rows(dataset, 3, seed=7)
    assert len(random_rows) == 3
    assert len({row.row_id for row in random_rows}) == 3


def test_dashboard_control_batch_inserts_only_when_invoked(tmp_path: Path) -> None:
    control_mod = _load_benchmark("bench_control_batch", "control.py")
    ds_mod = _load_benchmark("bench_dataset_batch", "dataset.py")
    cfg_mod = _load_benchmark("bench_config_batch", "config.py")
    store_mod = _load_benchmark("bench_store_batch", "duckdb_store.py")
    import benchmark.runner as shared_runner

    path = tmp_path / "data.jsonl"
    rows = [{"id": str(i), "prompt": f"p{i}", "max_tokens": 8} for i in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dataset = ds_mod.load_dataset(path)

    db_path = tmp_path / "batch.duckdb"
    store = store_mod.BenchmarkStore(db_path)
    store.insert_run(
        run_id="batch-run",
        dataset_path=str(path),
        dataset_hash=dataset.content_hash,
        concurrency_levels=[1],
        instance_type="gpu_1x_h100_sxm5",
        region="us-east-1",
        git_sha="deadbeef",
        archive_path=None,
        hourly_usd=3.29,
        node_count=1,
        status="ready",
    )

    progress: list[tuple[int, int, str]] = []

    async def fake_transport(**_kwargs):
        now = time.perf_counter()
        return [now, now + 0.001, now + 0.002], None

    async def _run_batch() -> None:
        shared_runner._stream_chat_override = fake_transport
        try:
            await control_mod.run_batch_for_nodes(
                config=cfg_mod.BenchmarkConfig(dataset_path=path),
                store=store,
                run_id="batch-run",
                nodes=[control_mod.NodeTarget(node_index=0, model_id="test-model", local_port=19191)],
                api_key="test",
                rows=dataset.rows,
                concurrency=1,
                progress=lambda _req, node, row, status: progress.append((node, row, status)),
            )
        finally:
            shared_runner._stream_chat_override = None

    assert store.conn.execute("SELECT count(*) FROM benchmark_requests").fetchone()[0] == 0
    asyncio.run(_run_batch())
    store.close()

    conn = duckdb.connect(str(db_path), read_only=True)
    assert conn.execute("SELECT count(*) FROM benchmark_requests").fetchone()[0] == 2
    assert conn.execute("SELECT status FROM benchmark_runs WHERE run_id = 'batch-run'").fetchone()[0] == "ready"
    conn.close()
    assert sorted(progress) == [(0, 0, "ok"), (0, 1, "ok")]


def test_concurrency_sweep_overlaps_requests(tmp_path: Path) -> None:
    runner_mod = _load_benchmark("bench_runner_conc", "runner.py")
    ds_mod = _load_benchmark("bench_dataset_conc", "dataset.py")
    store_mod = _load_benchmark("bench_store_conc", "duckdb_store.py")

    path = tmp_path / "data.jsonl"
    rows = [{"id": str(i), "prompt": f"p{i}", "max_tokens": 8} for i in range(4)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dataset = ds_mod.load_dataset(path)

    db_path = tmp_path / "conc.duckdb"
    store = store_mod.BenchmarkStore(db_path)

    in_flight = 0
    max_in_flight = 0
    delay_s = 0.08
    lock = asyncio.Lock()

    async def delayed_transport(**_kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(delay_s)
        async with lock:
            in_flight -= 1
        now = time.perf_counter()
        return [now, now + 0.001], None

    async def _run_sweep() -> float:
        runner_mod._stream_chat_override = delayed_transport
        try:
            t0 = time.perf_counter()
            await runner_mod.run_benchmark_for_node(
                base_url="http://127.0.0.1:19191/v1",
                api_key="test",
                model="test-model",
                dataset=dataset,
                concurrency_levels=(4,),
                default_max_tokens=8,
                store=store,
                run_id="conc-run",
                node_index=0,
            )
            return time.perf_counter() - t0
        finally:
            runner_mod._stream_chat_override = None

    elapsed = asyncio.run(_run_sweep())
    store.close()

    assert max_in_flight >= 2
    assert elapsed < delay_s * 4 * 0.85


def test_dry_run_mode(tmp_path: Path, monkeypatch) -> None:
    cfg = _load_benchmark("bench_config_dry", "config.py")
    run_state_mod = _load_benchmark("bench_run_state_dry", "run_state.py")
    store_mod = _load_benchmark("bench_store_dry", "duckdb_store.py")
    inst = _load_benchmark("bench_instance_dry", "instance.py")
    cli = _load_module("lambda_benchmark_dry", SCRIPTS / "lambda_benchmark.py")

    db_path = tmp_path / "dry.duckdb"
    runs_dir = tmp_path / "benchmark-runs"
    monkeypatch.setattr(cfg, "BENCHMARK_RUNS_DIR", runs_dir)
    monkeypatch.setattr(cfg, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(run_state_mod, "BENCHMARK_RUNS_DIR", runs_dir)
    import benchmark.run_state as shared_run_state

    monkeypatch.setattr(shared_run_state, "BENCHMARK_RUNS_DIR", runs_dir)

    def fake_regions(_token: str, instance_type: str) -> list[str]:
        if instance_type == "gpu_1x_h100_sxm5":
            return ["us-east-1"]
        return []

    monkeypatch.setattr(inst, "instance_regions_with_capacity", fake_regions)
    monkeypatch.setattr(inst, "pick_region", lambda *_a, **_k: "us-east-1")

    config = cli.BenchmarkConfig(db_path=db_path, dry_run=True)
    rc = cli.run_dry_run(config, select_capacity_fn=inst.select_instance_type_with_capacity)
    assert rc == 0
    assert db_path.exists()
    assert any(runs_dir.glob("*.json"))

    conn = duckdb.connect(str(db_path), read_only=True)
    row = conn.execute(
        "SELECT instance_type, region FROM benchmark_runs LIMIT 1"
    ).fetchone()
    conn.close()
    assert row == ("gpu_1x_h100_sxm5", "us-east-1")


@pytest.mark.skipif(
    not __import__("os").environ.get("RUN_LAMBDA_BENCHMARK_E2E"),
    reason="Set RUN_LAMBDA_BENCHMARK_E2E=1 to run real Lambda benchmark E2E",
)
def test_real_lambda_benchmark_e2e() -> None:
    cli = _load_module("lambda_benchmark_e2e", SCRIPTS / "lambda_benchmark.py")
    # Minimal dataset for cost control; launches, warms, then immediately cleans up.
    sample = ROOT / "benchmark" / "multi_model" / "sample_dataset.jsonl"
    config = cli.BenchmarkConfig(dataset_path=sample, concurrency_levels=(1,))
    rc = cli.run_benchmark(config, wait_for_control_fn=lambda _run_id, _db_path: None)
    assert rc == 0

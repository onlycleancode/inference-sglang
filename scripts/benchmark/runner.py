"""Benchmark workload execution against tunneled MiniSGLang nodes."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

from benchmark.cleanup import fetch_gpu_sample_via_ssh
from benchmark.config import BenchmarkConfig
from benchmark.dataset import (
    Dataset,
    DatasetRow,
    estimate_prompt_tokens,
    load_dataset,
    output_length_bucket,
    prompt_length_bucket,
    row_prompt_token_estimate,
    warmup_request_body,
)
from benchmark.duckdb_store import BenchmarkStore
from benchmark.metrics import compute_request_metrics
from benchmark.run_state import RunState

StreamChatFn = Callable[..., Awaitable[tuple[list[float], str | None]]]
RequestCompleteFn = Callable[[str, int, int, str], None]

# Optional override for unit tests (delayed fake transport).
_stream_chat_override: StreamChatFn | None = None


def _upstream_root(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/v1"):
        return value[:-3]
    return value


def _urlopen_sync(req: urllib.request.Request, *, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


async def wait_for_health(base_url: str, *, timeout_s: int = 900, interval_s: int = 5) -> None:
    deadline = time.time() + timeout_s
    url = f"{_upstream_root(base_url)}/health"
    while time.time() < deadline:
        try:
            resp = await asyncio.to_thread(_urlopen_sync, urllib.request.Request(url), timeout=5)
            with resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError):
            pass
        await asyncio.sleep(interval_s)
    raise RuntimeError(f"Timed out waiting for health at {base_url}")


async def wait_for_models(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout_s: int = 900,
    interval_s: int = 5,
) -> str:
    deadline = time.time() + timeout_s
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = await asyncio.to_thread(_urlopen_sync, req, timeout=5)
            with resp:
                payload = json.loads(resp.read().decode())
                models = payload.get("data") or payload.get("models") or []
                if models:
                    first = models[0]
                    return first.get("id") or first.get("name") or str(first)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError):
            pass
        await asyncio.sleep(interval_s)
    raise RuntimeError(f"Timed out waiting for /v1/models at {base_url}")


def _stream_chunk_carries_content(chunk: dict[str, Any]) -> bool:
    """True when an SSE chunk delivers decoded output (not a finish-only chunk)."""
    choices = chunk.get("choices") or []
    if not choices:
        return False
    choice = choices[0]
    if choice.get("finish_reason") is not None:
        return False
    delta = choice.get("delta") or {}
    return bool(delta.get("content"))


def _stream_chat_completion_sync(
    *,
    base_url: str,
    api_key: str,
    model: str,
    body: dict[str, Any],
) -> tuple[list[float], str | None]:
    """Blocking SSE chat completion; run via asyncio.to_thread from async callers."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {**body, "model": model, "stream": True}
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    tics = [time.perf_counter()]
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if _stream_chunk_carries_content(chunk):
                    tics.append(time.perf_counter())
    except Exception as exc:  # noqa: BLE001 - store error on request row
        return tics, str(exc)
    return tics, None


async def _stream_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    body: dict[str, Any],
) -> tuple[list[float], str | None]:
    """Return perf_counter timestamps for streamed chunks and optional error."""
    if _stream_chat_override is not None:
        return await _stream_chat_override(
            base_url=base_url,
            api_key=api_key,
            model=model,
            body=body,
        )
    return await asyncio.to_thread(
        _stream_chat_completion_sync,
        base_url=base_url,
        api_key=api_key,
        model=model,
        body=body,
    )


async def run_warmup(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompts: tuple[str, ...],
    default_max_tokens: int,
    store: BenchmarkStore,
    run_id: str,
    node_index: int,
) -> float:
    started = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        body = warmup_request_body(prompt)
        tics, err = await _stream_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            body=body,
        )
        request_id = f"{run_id}-warmup-{node_index}-{idx}"
        started_at = datetime.now(timezone.utc)
        store.insert_request(
            request_id=request_id,
            run_id=run_id,
            node_index=node_index,
            model_id=model,
            concurrency=1,
            dataset_row_id=f"warmup-{idx}",
            dataset_row_index=-1,
            status="error" if err else "ok",
            error_message=err,
            ttft_ms=None,
            tpot_ms=None,
            e2e_s=None,
            output_tokens=None,
            prompt_len_bucket=prompt_length_bucket(estimate_prompt_tokens(prompt)),
            output_len_bucket=None,
            output_tokens_per_sec=None,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            is_warmup=True,
        )
        if tics:
            store.insert_stream_tokens(request_id, tics)
    return time.perf_counter() - started


async def _run_one_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    row: DatasetRow,
    default_max_tokens: int,
) -> tuple[list[float], str | None]:
    body = row.request_body(default_max_tokens)
    return await _stream_chat_completion(base_url=base_url, api_key=api_key, model=model, body=body)


async def run_benchmark_for_node(
    *,
    base_url: str,
    api_key: str,
    model: str,
    dataset: Dataset,
    concurrency_levels: tuple[int, ...],
    default_max_tokens: int,
    store: BenchmarkStore,
    run_id: str,
    node_index: int,
    on_request_complete: RequestCompleteFn | None = None,
) -> None:
    for concurrency in concurrency_levels:
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(row: DatasetRow) -> None:
            async with sem:
                started_at = datetime.now(timezone.utc)
                t0 = time.perf_counter()
                tics, err = await _run_one_request(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    row=row,
                    default_max_tokens=default_max_tokens,
                )
                request_id = f"{run_id}-n{node_index}-c{concurrency}-{row.row_id}-{uuid.uuid4().hex[:6]}"
                prompt_token_count = row_prompt_token_estimate(row)
                if err or len(tics) < 2:
                    store.insert_request(
                        request_id=request_id,
                        run_id=run_id,
                        node_index=node_index,
                        model_id=model,
                        concurrency=concurrency,
                        dataset_row_id=row.row_id,
                        dataset_row_index=row.row_index,
                        status="error",
                        error_message=err or "insufficient stream timestamps",
                        ttft_ms=None,
                        tpot_ms=None,
                        e2e_s=time.perf_counter() - t0,
                        output_tokens=max(len(tics) - 1, 0),
                        prompt_len_bucket=prompt_length_bucket(prompt_token_count),
                        output_len_bucket=None,
                        output_tokens_per_sec=None,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                        is_warmup=False,
                    )
                    if tics:
                        store.insert_stream_tokens(request_id, tics)
                    if on_request_complete is not None:
                        on_request_complete(request_id, node_index, row.row_index, "error")
                    return

                metrics = compute_request_metrics(tics)
                store.insert_request(
                    request_id=request_id,
                    run_id=run_id,
                    node_index=node_index,
                    model_id=model,
                    concurrency=concurrency,
                    dataset_row_id=row.row_id,
                    dataset_row_index=row.row_index,
                    status="ok",
                    error_message=None,
                    ttft_ms=metrics.ttft_s * 1000.0,
                    tpot_ms=metrics.tpot_s * 1000.0,
                    e2e_s=metrics.e2e_s,
                    output_tokens=metrics.output_tokens,
                    prompt_len_bucket=prompt_length_bucket(prompt_token_count),
                    output_len_bucket=output_length_bucket(metrics.output_tokens),
                    output_tokens_per_sec=metrics.output_tokens_per_sec,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    is_warmup=False,
                )
                store.insert_stream_tokens(request_id, tics)
                if on_request_complete is not None:
                    on_request_complete(request_id, node_index, row.row_index, "ok")

        await asyncio.gather(*[_bounded(row) for row in dataset.rows])


async def sample_node_status(
    *,
    base_url: str,
    store: BenchmarkStore,
    run_id: str,
    node_index: int,
) -> None:
    url = f"{_upstream_root(base_url)}/status"
    try:
        resp = await asyncio.to_thread(_urlopen_sync, urllib.request.Request(url), timeout=5)
        with resp:
            payload = json.loads(resp.read().decode())
            store.insert_status_sample(run_id=run_id, node_index=node_index, payload=payload)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        pass


def sample_node_gpu(
    *,
    ip: str,
    ssh_key: Path,
    store: BenchmarkStore,
    run_id: str,
    node_index: int,
) -> None:
    sample = fetch_gpu_sample_via_ssh(ip, ssh_key)
    if sample is None:
        return
    memory_mb, util_pct = sample
    store.insert_gpu_sample(
        run_id=run_id,
        node_index=node_index,
        memory_mb=memory_mb,
        utilization_pct=util_pct,
    )


async def prepare_nodes_for_dashboard(
    *,
    config: BenchmarkConfig,
    run_state: RunState,
    store: BenchmarkStore,
    api_key: str,
    ssh_key: Path,
) -> Dataset:
    """Wait for nodes, run warmup, and stop before measured dataset inference."""
    dataset = load_dataset(config.dataset_path)
    run_state.dataset_path = str(config.dataset_path)
    run_state.dataset_hash = dataset.content_hash
    run_state.save()

    for node in run_state.nodes:
        base_url = f"http://127.0.0.1:{node.local_port}/v1"
        await wait_for_health(base_url)
        model = await wait_for_models(base_url, api_key=api_key)
        warmup_s = await run_warmup(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompts=config.warmup_prompts,
            default_max_tokens=config.default_max_tokens,
            store=store,
            run_id=run_state.run_id,
            node_index=node.node_index,
        )
        store.insert_node(
            run_id=run_state.run_id,
            node_index=node.node_index,
            model_id=node.model_id,
            instance_id=node.instance_id,
            ip=node.ip,
            local_port=node.local_port,
            warmup_s=warmup_s,
        )
        await sample_node_status(base_url=base_url, store=store, run_id=run_state.run_id, node_index=node.node_index)
        if node.ip:
            sample_node_gpu(ip=node.ip, ssh_key=ssh_key, store=store, run_id=run_state.run_id, node_index=node.node_index)
    return dataset


async def run_full_benchmark(
    *,
    config: BenchmarkConfig,
    run_state: RunState,
    store: BenchmarkStore,
    api_key: str,
    ssh_key: Path,
) -> None:
    dataset = await prepare_nodes_for_dashboard(
        config=config,
        run_state=run_state,
        store=store,
        api_key=api_key,
        ssh_key=ssh_key,
    )

    tasks = []
    for node in run_state.nodes:
        base_url = f"http://127.0.0.1:{node.local_port}/v1"
        model = await wait_for_models(base_url, api_key=api_key)
        tasks.append(
            run_benchmark_for_node(
                base_url=base_url,
                api_key=api_key,
                model=model,
                dataset=dataset,
                concurrency_levels=config.concurrency_levels,
                default_max_tokens=config.default_max_tokens,
                store=store,
                run_id=run_state.run_id,
                node_index=node.node_index,
            )
        )
    await asyncio.gather(*tasks)

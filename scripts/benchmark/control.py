"""Dashboard-controlled benchmark batch helpers."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark.config import BenchmarkConfig
from benchmark.dataset import Dataset, DatasetRow
from benchmark.duckdb_store import BenchmarkStore
from benchmark.runner import run_benchmark_for_node, sample_node_gpu, sample_node_status


@dataclass(frozen=True)
class NodeTarget:
    node_index: int
    model_id: str
    local_port: int
    ip: str | None = None


BatchProgressFn = Callable[[str, int, int, str], None]


def dataset_subset(dataset: Dataset, rows: Sequence[DatasetRow]) -> Dataset:
    return Dataset(path=dataset.path, rows=tuple(rows), content_hash=dataset.content_hash)


def select_next_rows(dataset: Dataset, next_row_index: int, count: int) -> tuple[tuple[DatasetRow, ...], int]:
    start = min(max(next_row_index, 0), dataset.size)
    stop = min(start + max(count, 0), dataset.size)
    return tuple(dataset.rows[start:stop]), stop


def select_random_rows(dataset: Dataset, count: int, *, seed: int | None = None) -> tuple[DatasetRow, ...]:
    if count <= 0:
        return ()
    rng = random.Random(seed)
    sample_size = min(count, dataset.size)
    return tuple(rng.sample(list(dataset.rows), sample_size))


async def run_batch_for_nodes(
    *,
    config: BenchmarkConfig,
    store: BenchmarkStore,
    run_id: str,
    nodes: Sequence[NodeTarget],
    api_key: str,
    rows: Sequence[DatasetRow],
    concurrency: int,
    progress: BatchProgressFn | None = None,
    ssh_key: Path | None = None,
) -> None:
    """Send one dashboard-selected row batch to each warm benchmark node."""
    if not rows or not nodes:
        return

    dataset = Dataset(path=config.dataset_path, rows=tuple(rows), content_hash="dashboard-batch")
    safe_concurrency = max(1, concurrency)
    store.update_run_status(run_id, "processing")
    try:
        tasks = []
        for node in nodes:
            base_url = f"http://127.0.0.1:{node.local_port}/v1"
            tasks.append(
                run_benchmark_for_node(
                    base_url=base_url,
                    api_key=api_key,
                    model=node.model_id,
                    dataset=dataset,
                    concurrency_levels=(safe_concurrency,),
                    default_max_tokens=config.default_max_tokens,
                    store=store,
                    run_id=run_id,
                    node_index=node.node_index,
                    on_request_complete=progress,
                )
            )
        await asyncio.gather(*tasks)

        for node in nodes:
            base_url = f"http://127.0.0.1:{node.local_port}/v1"
            await sample_node_status(base_url=base_url, store=store, run_id=run_id, node_index=node.node_index)
            if ssh_key is not None and node.ip:
                sample_node_gpu(ip=node.ip, ssh_key=ssh_key, store=store, run_id=run_id, node_index=node.node_index)
    finally:
        store.update_run_status(run_id, "ready")

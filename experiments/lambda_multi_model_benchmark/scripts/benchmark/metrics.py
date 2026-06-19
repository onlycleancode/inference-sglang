"""Benchmark metric calculations from streaming timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RequestMetrics:
    ttft_s: float
    tpot_s: float
    e2e_s: float
    output_tokens: int
    output_tokens_per_sec: float
    inter_token_latencies_s: tuple[float, ...]


def compute_request_metrics(tics: list[float]) -> RequestMetrics:
    """Compute TTFT, TPOT, E2E, and throughput from perf_counter timestamps."""
    if len(tics) < 2:
        raise ValueError("Need at least two timestamps for metrics")

    ttft_s = tics[1] - tics[0]
    e2e_s = tics[-1] - tics[0]
    deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
    decode_deltas = deltas[1:] if len(deltas) > 1 else []
    output_tokens = max(len(tics) - 1, 0)

    if decode_deltas:
        tpot_s = sum(decode_deltas) / len(decode_deltas)
    else:
        tpot_s = 0.0

    output_tokens_per_sec = output_tokens / e2e_s if e2e_s > 0 else 0.0
    return RequestMetrics(
        ttft_s=ttft_s,
        tpot_s=tpot_s,
        e2e_s=e2e_s,
        output_tokens=output_tokens,
        output_tokens_per_sec=output_tokens_per_sec,
        inter_token_latencies_s=tuple(decode_deltas),
    )


def percentile(values: Iterable[float], pct: float) -> float:
    """Return the pct percentile (0-100) using nearest-rank on sorted values."""
    sorted_vals = sorted(values)
    if not sorted_vals:
        return 0.0
    rank = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    rank = max(0, min(rank, len(sorted_vals) - 1))
    return sorted_vals[rank]


def summarize_latencies(values: Iterable[float]) -> dict[str, float]:
    vals = list(values)
    if not vals:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "avg": sum(vals) / len(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "max": max(vals),
    }


def request_throughput_per_s(
    *,
    request_count: int,
    wall_start_s: float,
    wall_end_s: float,
) -> float | None:
    """Request count divided by elapsed wall time for a concurrency group."""
    elapsed_s = wall_end_s - wall_start_s
    if request_count <= 0 or elapsed_s <= 0:
        return None
    return request_count / elapsed_s


def cost_per_million_output_tokens(
    total_output_tokens: int,
    benchmark_duration_s: float,
    hourly_usd: float,
    node_count: int,
) -> float | None:
    """Estimate USD per 1M output tokens from wall-clock benchmark duration."""
    if total_output_tokens <= 0 or benchmark_duration_s <= 0:
        return None
    total_cost = hourly_usd * node_count * (benchmark_duration_s / 3600.0)
    return total_cost / (total_output_tokens / 1_000_000.0)

#!/usr/bin/env python3
"""Streamlit dashboard for multi-model Lambda benchmark results and control."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from lambda_common import ENV_FILE, load_env_file

from benchmark.config import DEFAULT_DB_PATH, BenchmarkConfig
from benchmark.control import NodeTarget, run_batch_for_nodes, select_next_rows, select_random_rows
from benchmark.dataset import load_dataset
from benchmark.duckdb_store import BenchmarkStore
from benchmark.metrics import request_throughput_per_s


READY_STATUSES = {"ready"}


def _filter_requests(requests: pd.DataFrame, **filters: list) -> pd.DataFrame:
    """Apply sidebar filters; null output_len_bucket rows always pass bucket filter."""
    return requests[
        requests["model_id"].isin(filters["models"])
        & requests["concurrency"].isin(filters["concurrency"])
        & requests["prompt_len_bucket"].isin(filters["prompt_buckets"])
        & (
            requests["output_len_bucket"].isin(filters["output_buckets"])
            | requests["output_len_bucket"].isna()
        )
        & requests["status"].isin(filters["statuses"])
    ]


def _throughput_by_group(ok: pd.DataFrame) -> pd.DataFrame:
    """Request throughput = count / wall-clock span per model and concurrency."""
    grouped = (
        ok.groupby(["model_id", "concurrency"])
        .agg(
            req_count=("request_id", "count"),
            wall_start=("started_at", "min"),
            wall_end=("finished_at", "max"),
        )
        .reset_index()
    )
    elapsed_s = (
        pd.to_datetime(grouped["wall_end"]) - pd.to_datetime(grouped["wall_start"])
    ).dt.total_seconds()
    grouped["req_per_s"] = [
        request_throughput_per_s(
            request_count=int(row.req_count),
            wall_start_s=0.0,
            wall_end_s=float(elapsed),
        )
        or 0.0
        for row, elapsed in zip(grouped.itertuples(), elapsed_s, strict=True)
    ]
    return grouped


def _load_df(conn: duckdb.DuckDBPyConnection, query: str, params: list | None = None) -> pd.DataFrame:
    return conn.execute(query, params or []).df()


def _node_targets(nodes: pd.DataFrame) -> list[NodeTarget]:
    targets: list[NodeTarget] = []
    for row in nodes.itertuples():
        targets.append(
            NodeTarget(
                node_index=int(row.node_index),
                model_id=str(row.model_id),
                local_port=int(row.local_port),
                ip=None if pd.isna(row.ip) else str(row.ip),
            )
        )
    return targets


def _run_control_batch(
    *,
    db_path: Path,
    run_id: str,
    dataset_path: Path,
    nodes: pd.DataFrame,
    api_key: str,
    count: int,
    concurrency: int,
    mode: str,
) -> None:
    dataset = load_dataset(dataset_path)
    store = BenchmarkStore(db_path)
    try:
        if mode == "next":
            cursor = store.get_next_row_index(run_id)
            rows, next_cursor = select_next_rows(dataset, cursor, count)
        else:
            rows = select_random_rows(dataset, count)
            next_cursor = None

        if not rows:
            st.warning("No prompts selected. The sequential cursor is at the end of the dataset.")
            return

        total = len(rows) * len(nodes)
        completed = 0
        progress_bar = st.progress(0, text=f"Sending 0/{total} requests")
        status_box = st.empty()
        metrics_box = st.empty()
        live_rows: list[tuple[float | None, float | None, float | None, str]] = []

        def progress(request_id: str, node_index: int, row_index: int, status: str) -> None:
            nonlocal completed
            completed += 1
            progress_bar.progress(
                min(completed / total, 1.0),
                text=f"Sending {completed}/{total} requests",
            )
            status_box.caption(f"Last: node {node_index}, row {row_index}, {status}")
            metric_row = store.conn.execute(
                """
                SELECT ttft_ms, tpot_ms, e2e_s, status
                FROM benchmark_requests
                WHERE request_id = ?
                """,
                [request_id],
            ).fetchone()
            if metric_row is not None:
                live_rows.append(metric_row)
                ok_rows = [row for row in live_rows if row[3] == "ok"]
                err_count = len(live_rows) - len(ok_rows)
                avg_ttft = sum(row[0] for row in ok_rows if row[0] is not None) / len(ok_rows) if ok_rows else 0.0
                avg_tpot = sum(row[1] for row in ok_rows if row[1] is not None) / len(ok_rows) if ok_rows else 0.0
                avg_e2e = sum(row[2] for row in ok_rows if row[2] is not None) / len(ok_rows) if ok_rows else 0.0
                metrics_box.caption(
                    f"Batch metrics: ok {len(ok_rows)}, errors {err_count}, "
                    f"avg TTFT {avg_ttft:.1f} ms, avg TPOT {avg_tpot:.1f} ms, avg E2E {avg_e2e:.2f} s"
                )

        config = BenchmarkConfig(dataset_path=dataset_path)
        asyncio.run(
            run_batch_for_nodes(
                config=config,
                store=store,
                run_id=run_id,
                nodes=_node_targets(nodes),
                api_key=api_key,
                rows=rows,
                concurrency=concurrency,
                progress=progress,
            )
        )
        if next_cursor is not None:
            store.set_next_row_index(run_id, next_cursor)
    finally:
        store.close()

    st.success(f"Sent {len(rows)} prompts to each warm node.")
    st.rerun()


def _render_prompt_controls(db_path: Path, run_meta: pd.Series, nodes: pd.DataFrame) -> None:
    run_id = str(run_meta["run_id"])
    status = str(run_meta["status"])
    dataset_path = Path(str(run_meta["dataset_path"]))

    st.subheader("Prompt Control")
    if nodes.empty:
        st.info("No node metadata found yet.")
        return
    if not dataset_path.exists():
        st.warning(f"Dataset not found: {dataset_path}")
        return

    dataset = load_dataset(dataset_path)
    cursor = 0
    if status in READY_STATUSES:
        store = BenchmarkStore(db_path)
        try:
            cursor = store.get_next_row_index(run_id)
        finally:
            store.close()

    api_key_default = os.getenv("MINISGL_API_KEY", "")
    api_key = st.text_input("MiniSGLang API key", value=api_key_default, type="password")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        count = st.number_input("Prompts per model", min_value=1, max_value=dataset.size, value=min(10, dataset.size))
    with col2:
        concurrency = st.number_input("Concurrency", min_value=1, max_value=128, value=1)
    with col3:
        st.metric("Next row", f"{min(cursor, dataset.size)}/{dataset.size}")
    with col4:
        st.caption(f"Dataset: {dataset_path.name}")
        st.caption("Buttons send prompts to every warm model node; nothing is sent automatically.")

    disabled = status not in READY_STATUSES or not api_key
    if status == "processing":
        st.warning("A dashboard batch is already marked as processing.")
    elif status not in READY_STATUSES:
        st.info(f"Controls enable after nodes are warm. Current run status: {status}")
    elif not api_key:
        st.warning("Enter the MiniSGLang API key to send prompts.")

    send_next, send_random = st.columns(2)
    with send_next:
        if st.button("Send Next Prompts", disabled=disabled, use_container_width=True):
            _run_control_batch(
                db_path=db_path,
                run_id=run_id,
                dataset_path=dataset_path,
                nodes=nodes,
                api_key=api_key,
                count=int(count),
                concurrency=int(concurrency),
                mode="next",
            )
    with send_random:
        if st.button("Process Random Prompts", disabled=disabled, use_container_width=True):
            _run_control_batch(
                db_path=db_path,
                run_id=run_id,
                dataset_path=dataset_path,
                nodes=nodes,
                api_key=api_key,
                count=int(count),
                concurrency=int(concurrency),
                mode="random",
            )


def _sidebar_filters(requests: pd.DataFrame) -> dict[str, list]:
    models = sorted(requests["model_id"].dropna().unique().tolist())
    concurrencies = sorted(requests["concurrency"].dropna().unique().tolist())
    prompt_buckets = sorted(requests["prompt_len_bucket"].dropna().unique().tolist())
    output_buckets = sorted(requests["output_len_bucket"].dropna().unique().tolist())
    statuses = sorted(requests["status"].dropna().unique().tolist())
    return {
        "models": st.sidebar.multiselect("Model", models, default=models),
        "concurrency": st.sidebar.multiselect("Concurrency", concurrencies, default=concurrencies),
        "prompt_buckets": st.sidebar.multiselect("Prompt length bucket", prompt_buckets, default=prompt_buckets),
        "output_buckets": st.sidebar.multiselect("Output length bucket", output_buckets, default=output_buckets),
        "statuses": st.sidebar.multiselect("Status", statuses, default=statuses),
    }


def _render_charts(conn: duckdb.DuckDBPyConnection, run_id: str, filtered: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    ok = filtered[filtered["status"] == "ok"]

    with col1:
        if not ok.empty:
            ttft_summary = (
                ok.groupby("model_id")["ttft_ms"]
                .agg(["mean", lambda s: s.quantile(0.5), lambda s: s.quantile(0.9), lambda s: s.quantile(0.99)])
                .reset_index()
            )
            ttft_summary.columns = ["model_id", "avg", "p50", "p90", "p99"]
            fig = px.bar(
                ttft_summary.melt(id_vars="model_id", var_name="percentile", value_name="ttft_ms"),
                x="model_id",
                y="ttft_ms",
                color="percentile",
                barmode="group",
                title="TTFT percentiles by model (ms)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not ok.empty:
            fig = px.histogram(ok, x="tpot_ms", color="model_id", nbins=40, title="TPOT distribution (ms)")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if not ok.empty:
            tp = _throughput_by_group(ok)
            fig = px.line(
                tp,
                x="concurrency",
                y="req_per_s",
                color="model_id",
                markers=True,
                title="Request throughput vs concurrency (req/s)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if not ok.empty:
            fig = px.ecdf(ok, x="e2e_s", color="model_id", title="E2E latency CDF (s)")
            st.plotly_chart(fig, use_container_width=True)

    gpu = conn.execute(
        "SELECT * FROM benchmark_gpu_samples WHERE run_id = ? ORDER BY sampled_at",
        [run_id],
    ).df()
    if not gpu.empty:
        col5, col6 = st.columns(2)
        with col5:
            fig = px.line(
                gpu,
                x="sampled_at",
                y="memory_mb",
                color=gpu["node_index"].astype(str),
                title="GPU memory over time (MB)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col6:
            fig = px.line(
                gpu,
                x="sampled_at",
                y="utilization_pct",
                color=gpu["node_index"].astype(str),
                title="GPU utilization over time (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

    err = filtered.groupby("model_id")["status"].apply(lambda s: (s != "ok").mean()).reset_index(name="error_rate")
    if not err.empty:
        fig = px.bar(err, x="model_id", y="error_rate", title="Error rate by model")
        st.plotly_chart(fig, use_container_width=True)

    run_meta = conn.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", [run_id]).df()
    if not run_meta.empty and not ok.empty:
        hourly = float(run_meta.iloc[0].get("hourly_usd") or 0)
        node_count = int(run_meta.iloc[0].get("node_count") or 1)
        start = pd.to_datetime(run_meta.iloc[0]["started_at"], utc=True)
        raw_end = run_meta.iloc[0].get("finished_at")
        end = pd.to_datetime(raw_end, utc=True) if pd.notna(raw_end) else pd.Timestamp.now(tz="UTC")
        duration_s = (end - start).total_seconds()
        total_tokens = ok["output_tokens"].sum()
        if hourly > 0 and duration_s > 0 and total_tokens > 0:
            cost = hourly * node_count * (duration_s / 3600.0) / (total_tokens / 1_000_000.0)
            st.metric("Estimated cost per 1M output tokens (USD)", f"{cost:.2f}")


def main() -> None:
    st.set_page_config(page_title="MiniSGLang Multi-Model Benchmark", layout="wide")
    st.title("MiniSGLang Multi-Model Benchmark Dashboard")
    load_env_file(ENV_FILE)

    db_path = st.sidebar.text_input("DuckDB path", value=str(DEFAULT_DB_PATH))
    path = Path(db_path)
    if not path.exists():
        st.warning(f"Database not found: {path}")
        st.stop()

    meta_conn = duckdb.connect(str(path), read_only=True)
    runs = _load_df(meta_conn, "SELECT run_id, status, started_at FROM benchmark_runs ORDER BY started_at DESC")
    if runs.empty:
        st.info("No benchmark runs found.")
        meta_conn.close()
        return

    run_id = st.sidebar.selectbox("Run ID", runs["run_id"].tolist())
    run_meta_df = _load_df(meta_conn, "SELECT * FROM benchmark_runs WHERE run_id = ?", [run_id])
    nodes = _load_df(meta_conn, "SELECT * FROM benchmark_nodes WHERE run_id = ? ORDER BY node_index", [run_id])
    meta_conn.close()

    _render_prompt_controls(path, run_meta_df.iloc[0], nodes)

    conn = duckdb.connect(str(path), read_only=True)
    requests = conn.execute(
        """
        SELECT * FROM benchmark_requests
        WHERE run_id = ? AND is_warmup = FALSE
        """,
        [run_id],
    ).df()
    if requests.empty:
        st.info("No measured request rows yet. Use the prompt controls above to send a batch.")
        conn.close()
        return

    filters = _sidebar_filters(requests)
    filtered = _filter_requests(requests, **filters)
    _render_charts(conn, run_id, filtered)

    with st.expander("Raw request drilldown"):
        st.dataframe(filtered.sort_values(["model_id", "concurrency", "dataset_row_index"]))

    conn.close()


if __name__ == "__main__":
    main()

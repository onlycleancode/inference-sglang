"""DuckDB persistence for multi-model benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS benchmark_runs (
        run_id VARCHAR PRIMARY KEY,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        status VARCHAR,
        dataset_path VARCHAR,
        dataset_hash VARCHAR,
        concurrency_levels VARCHAR,
        instance_type VARCHAR,
        region VARCHAR,
        git_sha VARCHAR,
        archive_path VARCHAR,
        hourly_usd DOUBLE,
        node_count INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_nodes (
        run_id VARCHAR,
        node_index INTEGER,
        model_id VARCHAR,
        instance_id VARCHAR,
        ip VARCHAR,
        local_port INTEGER,
        model_load_s DOUBLE,
        warmup_s DOUBLE,
        PRIMARY KEY (run_id, node_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_requests (
        request_id VARCHAR PRIMARY KEY,
        run_id VARCHAR,
        node_index INTEGER,
        model_id VARCHAR,
        concurrency INTEGER,
        dataset_row_id VARCHAR,
        dataset_row_index INTEGER,
        status VARCHAR,
        error_message VARCHAR,
        ttft_ms DOUBLE,
        tpot_ms DOUBLE,
        e2e_s DOUBLE,
        output_tokens INTEGER,
        prompt_len_bucket VARCHAR,
        output_len_bucket VARCHAR,
        output_tokens_per_sec DOUBLE,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        is_warmup BOOLEAN DEFAULT FALSE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_stream_tokens (
        request_id VARCHAR,
        token_index INTEGER,
        timestamp_s DOUBLE,
        PRIMARY KEY (request_id, token_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_status_samples (
        run_id VARCHAR,
        node_index INTEGER,
        sampled_at TIMESTAMP,
        payload_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_gpu_samples (
        run_id VARCHAR,
        node_index INTEGER,
        sampled_at TIMESTAMP,
        memory_mb DOUBLE,
        utilization_pct DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_errors (
        run_id VARCHAR,
        node_index INTEGER,
        error_at TIMESTAMP,
        phase VARCHAR,
        message VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_cleanup (
        run_id VARCHAR PRIMARY KEY,
        cleaned_at TIMESTAMP,
        instance_ids VARCHAR,
        success BOOLEAN,
        remaining_instances VARCHAR,
        message VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS benchmark_control (
        run_id VARCHAR PRIMARY KEY,
        next_row_index INTEGER,
        updated_at TIMESTAMP
    )
    """,
)


class BenchmarkStore:
    """Thin wrapper around DuckDB inserts and analytics queries."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self.init_schema()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self._conn.execute(statement)

    def insert_run(
        self,
        *,
        run_id: str,
        dataset_path: str,
        dataset_hash: str,
        concurrency_levels: list[int],
        instance_type: str | None,
        region: str | None,
        git_sha: str | None,
        archive_path: str | None,
        hourly_usd: float | None,
        node_count: int,
        status: str = "running",
    ) -> None:
        now = datetime.now(timezone.utc)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_runs (
                run_id, started_at, finished_at, status, dataset_path, dataset_hash,
                concurrency_levels, instance_type, region, git_sha, archive_path,
                hourly_usd, node_count
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                now,
                status,
                dataset_path,
                dataset_hash,
                ",".join(str(v) for v in concurrency_levels),
                instance_type,
                region,
                git_sha,
                archive_path,
                hourly_usd,
                node_count,
            ],
        )

    def finish_run(self, run_id: str, status: str) -> None:
        self._conn.execute(
            """
            UPDATE benchmark_runs
            SET finished_at = ?, status = ?
            WHERE run_id = ?
            """,
            [datetime.now(timezone.utc), status, run_id],
        )

    def update_run_status(self, run_id: str, status: str) -> None:
        self._conn.execute(
            """
            UPDATE benchmark_runs
            SET status = ?
            WHERE run_id = ?
            """,
            [status, run_id],
        )

    def get_next_row_index(self, run_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT next_row_index
            FROM benchmark_control
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)

    def set_next_row_index(self, run_id: str, next_row_index: int) -> None:
        self._conn.execute(
            """
            INSERT INTO benchmark_control (run_id, next_row_index, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                next_row_index = excluded.next_row_index,
                updated_at = excluded.updated_at
            """,
            [run_id, next_row_index, datetime.now(timezone.utc)],
        )

    def insert_node(
        self,
        *,
        run_id: str,
        node_index: int,
        model_id: str,
        instance_id: str | None,
        ip: str | None,
        local_port: int,
        model_load_s: float | None = None,
        warmup_s: float | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO benchmark_nodes (
                run_id, node_index, model_id, instance_id, ip, local_port,
                model_load_s, warmup_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, node_index) DO UPDATE SET
                model_id = excluded.model_id,
                instance_id = excluded.instance_id,
                ip = excluded.ip,
                local_port = excluded.local_port,
                model_load_s = COALESCE(excluded.model_load_s, benchmark_nodes.model_load_s),
                warmup_s = COALESCE(excluded.warmup_s, benchmark_nodes.warmup_s)
            """,
            [
                run_id,
                node_index,
                model_id,
                instance_id,
                ip,
                local_port,
                model_load_s,
                warmup_s,
            ],
        )

    def insert_request(
        self,
        *,
        request_id: str,
        run_id: str,
        node_index: int,
        model_id: str,
        concurrency: int,
        dataset_row_id: str,
        dataset_row_index: int,
        status: str,
        error_message: str | None,
        ttft_ms: float | None,
        tpot_ms: float | None,
        e2e_s: float | None,
        output_tokens: int | None,
        prompt_len_bucket: str | None,
        output_len_bucket: str | None,
        output_tokens_per_sec: float | None,
        started_at: datetime,
        finished_at: datetime | None,
        is_warmup: bool = False,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_requests (
                request_id, run_id, node_index, model_id, concurrency,
                dataset_row_id, dataset_row_index, status, error_message,
                ttft_ms, tpot_ms, e2e_s, output_tokens, prompt_len_bucket,
                output_len_bucket, output_tokens_per_sec, started_at, finished_at,
                is_warmup
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                request_id,
                run_id,
                node_index,
                model_id,
                concurrency,
                dataset_row_id,
                dataset_row_index,
                status,
                error_message,
                ttft_ms,
                tpot_ms,
                e2e_s,
                output_tokens,
                prompt_len_bucket,
                output_len_bucket,
                output_tokens_per_sec,
                started_at,
                finished_at,
                is_warmup,
            ],
        )

    def insert_stream_tokens(self, request_id: str, tics: list[float]) -> None:
        if len(tics) < 2:
            return
        base = tics[0]
        rows = [(request_id, idx, ts - base) for idx, ts in enumerate(tics)]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO benchmark_stream_tokens
            (request_id, token_index, timestamp_s)
            VALUES (?, ?, ?)
            """,
            rows,
        )

    def insert_status_sample(
        self,
        *,
        run_id: str,
        node_index: int,
        payload: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO benchmark_status_samples (run_id, node_index, sampled_at, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            [run_id, node_index, datetime.now(timezone.utc), json.dumps(payload)],
        )

    def insert_gpu_sample(
        self,
        *,
        run_id: str,
        node_index: int,
        memory_mb: float,
        utilization_pct: float,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO benchmark_gpu_samples
            (run_id, node_index, sampled_at, memory_mb, utilization_pct)
            VALUES (?, ?, ?, ?, ?)
            """,
            [run_id, node_index, datetime.now(timezone.utc), memory_mb, utilization_pct],
        )

    def insert_error(
        self,
        *,
        run_id: str,
        node_index: int | None,
        phase: str,
        message: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO benchmark_errors (run_id, node_index, error_at, phase, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            [run_id, node_index, datetime.now(timezone.utc), phase, message],
        )

    def insert_cleanup(
        self,
        *,
        run_id: str,
        instance_ids: list[str],
        success: bool,
        remaining_instances: list[str],
        message: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_cleanup (
                run_id, cleaned_at, instance_ids, success, remaining_instances, message
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                datetime.now(timezone.utc),
                ",".join(instance_ids),
                success,
                ",".join(remaining_instances),
                message,
            ],
        )

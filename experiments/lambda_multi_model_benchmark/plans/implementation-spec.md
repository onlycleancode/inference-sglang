# Multi-Model Lambda Benchmark — Implementation Spec

This document records the architecture, technical decisions, and operational details for the multi-model MiniSGLang inference benchmark harness implemented per `plans/plan.md`.

## Architecture Overview

The harness launches **three same-type Lambda GPU instances** in one region, deploys MiniSGLang with identical server configuration on each node (only `MINISGL_MODEL` differs), exposes servers through **local SSH tunnels** on ports **19191–19193**, warms each node, and then stops before measured dataset inference. The Streamlit dashboard sends user-controlled batches from the Alpaca 2k JSONL workload, persists telemetry to **DuckDB**, and updates metrics as rows are recorded. The launcher process owns tunnels and cleanup; interrupting it terminates tracked instances.

```
┌─────────────────────┐     launch x3      ┌──────────────────────────┐
│ scripts/            │ ─────────────────► │ Lambda GPU instances     │
│ lambda_benchmark.py │                    │ (same type + region)     │
└─────────┬───────────┘                    └────────────┬─────────────┘
          │                                             │ SSH deploy
          │ tunnels 19191-19193                         ▼
          ▼                                  ┌──────────────────────────┐
┌─────────────────────┐   HTTP /v1           │ MiniSGLang (Docker)      │
│ dashboard.py        │ ◄─────────────────── │ model A / B / C          │
│ control.py          │                      └──────────────────────────┘
└─────────┬───────────┘
          │ inserts
          ▼
┌─────────────────────┐     Streamlit      ┌──────────────────────────┐
│ .lambda-runtime/    │ ◄───────────────── │ scripts/benchmark/       │
│ benchmark.duckdb    │                    │ dashboard.py             │
└─────────────────────┘                    └──────────────────────────┘
```

### Module Layout

| Path | Responsibility |
|------|----------------|
| `scripts/lambda_benchmark.py` | CLI entry point (`--dry-run`, dataset, concurrency, region) |
| `scripts/benchmark/config.py` | Model matrix, tunnel ports, server args, hourly cost table |
| `scripts/benchmark/models.py` | Model matrix validation |
| `scripts/benchmark/instance.py` | Capacity selection, 3-node cluster launch, IP polling |
| `scripts/benchmark/deploy.py` | Repo packaging, remote Docker deploy with parity config |
| `scripts/benchmark/cleanup.py` | `BenchmarkCleanupManager`, tunnels, termination, verification |
| `scripts/benchmark/run_state.py` | Run JSON under `.lambda-runtime/benchmark-runs/<run_id>.json` |
| `scripts/benchmark/dataset.py` | JSONL parsing, stable SHA-256 hash, length buckets |
| `scripts/benchmark/metrics.py` | TTFT, TPOT, E2E, percentiles, cost estimation |
| `scripts/benchmark/duckdb_store.py` | Schema creation and typed inserts |
| `scripts/benchmark/runner.py` | Health/models wait, warmup, benchmark sweep, GPU/status sampling |
| `scripts/benchmark/control.py` | Dashboard batch row selection and per-node batch execution |
| `scripts/benchmark/dashboard.py` | Streamlit + Plotly analytics and prompt control UI |
| `benchmark/multi_model/alpaca_2k_prompts.jsonl` | Default 2,000-row Alpaca prompt workload converted from `mhenrichsen/alpaca_2k_test` |
| `benchmark/multi_model/sample_dataset.jsonl` | Sample workload for dry-run and local testing |
| `tests/scripts/test_lambda_benchmark.py` | Unit tests + gated E2E |

## Technical Decisions

### Instance selection

**Decision:** Reuse `lambda_common.resolve_instance_types()` ordering (`gpu_1x_h100_sxm5` → `gpu_1x_h100_pcie` → A100 fallbacks) and add `select_instance_type_with_capacity()` that queries `/instance-types` via existing `instance_regions_with_capacity()`.

**Why:** Keeps parity with single-instance deploy behavior while adding a benchmark-specific “pick first type with any region capacity” step before launching all three nodes together.

**Launch:** `launch_benchmark_cluster()` sends one `/instance-operations/launch` with `quantity: 3` and name prefix `minisgl-benchmark-<run_id>`. On partial response, it immediately terminates any returned IDs before raising.

### Run state vs `.env`

**Decision:** Store orchestration state in `.lambda-runtime/benchmark-runs/<run_id>.json` (already gitignored via `.lambda-runtime/`). Do **not** write `LAMBDA_PUBLIC_IP` to root `.env`.

**Why:** Three nodes cannot share a single IP field; run-scoped JSON preserves node index, model, instance ID, IP, tunnel PID, and status for debugging/resume visibility.

### Deploy parity

**Decision:** Reuse `scripts/package_lambda.sh` tarball and mirror `lambda_deploy.remote_deploy()` patterns. Benchmark deploy writes `deploy/lambda/.env` plus `benchmark-compose.override.yml` with fixed server args:

- `--dtype bfloat16`
- `--cache-type radix`
- `--page-size 64`
- `--max-seq-len-override 32768`
- `--cuda-graph-max-bs 64`

Only `MINISGL_MODEL` differs per node. Same archive, git SHA (via `git rev-parse HEAD`), `MINISGL_API_KEY`, and `HF_TOKEN` across nodes.

**Why:** Ensures measured differences reflect model architecture/size, not config drift.

### Tunnel ports

**Decision:** Remote port **1919** on each VM; local ports **19191**, **19192**, **19193** map to nodes 0–2.

**Why:** Avoids collision with existing single-instance tunnel default (`1919`) and gives stable, documented local URLs for parallel benchmarking.

### Workload parity

**Decision:**

1. Wait for `/health` then `/v1/models` on each tunnel.
2. Run identical warmup prompts (`DEFAULT_WARMUP_PROMPTS`); store rows with `is_warmup=TRUE` but exclude from dashboard aggregates.
3. Mark the run `ready`, close the launcher DuckDB write connection, and keep the process alive so SSH tunnels stay open.
4. Parse JSONL dataset with required `prompt` **or** `messages`, optional `id`, `max_tokens`, metadata. Default: `benchmark/multi_model/alpaca_2k_prompts.jsonl` from [mhenrichsen/alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test).
5. Send measured requests only after the user presses a dashboard control:
   - **Send Next Prompts** uses the persisted `benchmark_control.next_row_index` cursor.
   - **Process Random Prompts** samples rows without moving the sequential cursor.
   - The selected prompt count is per model node; each batch is sent to every warm node at the dashboard-selected concurrency.

Sampling: `temperature=0.0`, top-level `ignore_eos=True`, `top_k=1` (aligned with MiniSGLang `OpenAICompletionRequest` fields).

**Why:** Warmup separates cold-cache effects from measured requests; identical order/concurrency enables fair cross-model comparison.

### DuckDB schema

**Decision:** Eight tables:

| Table | Contents |
|-------|----------|
| `benchmark_runs` | Run metadata, dataset hash, instance type/region, git SHA, hourly USD |
| `benchmark_nodes` | Per-node model, instance ID, IP, port, model load & warmup duration |
| `benchmark_requests` | Request-level TTFT/TPOT/E2E, buckets, status, concurrency |
| `benchmark_stream_tokens` | Normalized per-token timestamps (seconds from first tic) |
| `benchmark_status_samples` | `/status` JSON snapshots |
| `benchmark_gpu_samples` | `nvidia-smi` memory/utilization via SSH |
| `benchmark_errors` | Orchestration/request phase errors |
| `benchmark_cleanup` | Termination outcome and remaining instance IDs |
| `benchmark_control` | Dashboard sequential prompt cursor (`next_row_index`) |

Default DB path: `.lambda-runtime/benchmark.duckdb`.

At run start, the launcher archives any existing default dashboard DB files to
`.lambda-runtime/benchmark-db-archive/<run_id>/` before opening DuckDB. This
keeps historical telemetry available while ensuring a new dashboard session
does not display stale rows from earlier iterations.

**Why:** DuckDB supports local analytics, fast aggregations, and Streamlit/Pandas integration without a separate server.

### Metric formulas

From streaming `perf_counter` timestamps `tics[0..N]`:

| Metric | Formula |
|--------|---------|
| **TTFT** | `tics[1] - tics[0]` (stored as ms) |
| **TPOT** | Mean of inter-token deltas after first token: `mean(tics[i+1]-tics[i] for i>=1)` |
| **E2E** | `tics[-1] - tics[0]` |
| **Output tokens** | `len(tics) - 1` (one timestamp per streamed chunk) |
| **Output tok/s** | `output_tokens / E2E` |
| **Request throughput** | Computed in dashboard as count/duration by concurrency |
| **Percentiles** | Nearest-rank on sorted samples (p50/p90/p95/p99 helpers in `metrics.py`) |
| **Error rate** | `count(status != 'ok') / count(*)` per model |
| **Model load time** | Remote deploy wall time (includes Docker build + health wait); stored separately from request latency |
| **Cost / 1M output tokens** | `(hourly_usd × node_count × duration_hours) / (total_output_tokens / 1e6)` |

Warmup and model-download time are **excluded** from request latency metrics but stored in `benchmark_nodes`.

Hourly USD uses `INSTANCE_HOURLY_USD` lookup when Lambda API does not expose pricing.

### Cleanup strategy

**Decision:** `BenchmarkCleanupManager` context manager with:

1. `try/finally`-style `__exit__` always calling `cleanup()`
2. SIGINT/SIGTERM handlers that stop tunnels, terminate tracked IDs, write `benchmark_cleanup`
3. Post-terminate poll of `/instances` for up to 120s; **raise** if any `minisgl-benchmark*` instance remains `active` or `booting`

Injected `terminate_fn` / `list_instances_fn` enable unit tests without API calls.

During the dashboard-ready phase, the launcher closes its long-lived DuckDB connection so Streamlit can write request rows. Cleanup telemetry can reopen the DB by path and is best-effort if another dashboard process owns the DB lock; instance termination remains mandatory.

**Why:** Plan mandates no leaked benchmark instances; loud failure surfaces incomplete termination.

### Reuse of existing Lambda scripts

| Existing | Benchmark usage |
|----------|-----------------|
| `lambda_common.api_request` | All Lambda HTTP |
| `lambda_common.resolve_instance_types` | Instance type ordering |
| `lambda_common.instance_regions_with_capacity` | Capacity queries |
| `lambda_common.pick_region` | Region selection |
| `lambda_common.terminate_instance` | Cleanup |
| `lambda_common.list_instances` | Post-cleanup verification |
| `lambda_common.load_env_file` / `require` | Credential loading |
| `lambda_deploy` patterns | SSH/SCP, `.env` layout, Docker compose flow |
| `scripts/package_lambda.sh` | Identical repo archive |

No changes were required to `lambda_common.py` itself; benchmark modules compose existing helpers.

## Test Coverage Summary

| Test | Coverage |
|------|----------|
| `test_select_instance_type_prefers_h100` | H100 preferred when capacity exists |
| `test_select_instance_type_falls_back` | A100 fallback when H100 empty |
| `test_select_instance_type_no_capacity_raises` | Hard failure with no capacity |
| `test_cleanup_partial_launch_*` | Tracked IDs terminated |
| `test_cleanup_deploy_failure_*` | Exception during deploy still cleans up |
| `test_cleanup_benchmark_failure_*` | Benchmark exception path |
| `test_cleanup_keyboard_interrupt_*` | SIGINT handler terminates |
| `test_duckdb_schema_and_inserts` | Temp DB schema + CRUD |
| `test_dataset_parsing_*` | prompt/messages, missing max_tokens, bad JSONL |
| `test_dataset_hash_is_stable` | Deterministic dataset hash |
| `test_metric_calculations_*` | TTFT/TPOT/E2E from synthetic tics |
| `test_dry_run_mode` | Capacity validation path, run state, DuckDB init |
| `test_dataset_request_body_uses_top_level_sampling_fields` | Top-level `ignore_eos`/`top_k` in dataset and warmup bodies |
| `test_node_upsert_preserves_model_load_s_on_warmup_update` | Deploy then warmup node writes preserve `model_load_s` |
| `test_request_throughput_uses_wall_clock` | Throughput = count / wall time |
| `test_dashboard_throughput_uses_timedelta_wall_clock` | Dashboard throughput uses timedelta, not int64 ns |
| `test_stream_completion_skips_finish_reason_stop_chunk` | Finish-only SSE chunks excluded from token count |
| `test_dashboard_filter_keeps_null_output_bucket_errors` | Error rows with null output bucket visible |
| `test_concurrency_sweep_overlaps_requests` | Fake transport proves concurrent overlap |
| `test_default_dataset_is_alpaca_2k` | Default workload is the 2,000-row Alpaca JSONL |
| `test_dashboard_control_selects_next_and_random_rows` | Dashboard next/random row selection semantics |
| `test_dashboard_control_batch_inserts_only_when_invoked` | Dashboard batch helper records requests only when called |
| `test_real_lambda_benchmark_e2e` | Skipped unless `RUN_LAMBDA_BENCHMARK_E2E=1`; warms nodes then returns through injected wait hook |

**Result:** 22 passed, 1 skipped (E2E gate), 0 failed (verified locally after Alpaca/dashboard-control changes).

## How to Run

### Install optional dependencies

```bash
pip install -e ".[benchmark]"
# or: pip install duckdb streamlit plotly pandas pyarrow
```

### Dry-run (no Lambda)

```bash
python scripts/lambda_benchmark.py --dry-run
```

Validates model matrix, writes run state JSON, initializes DuckDB schema.

### Unit tests

```bash
pytest tests/scripts/test_lambda_benchmark.py -o addopts=
```

### Real benchmark

Requires `.env` with `LAMBDA_CLOUD_API_KEY`, `MINISGL_API_KEY`, `HF_TOKEN`, and registered SSH key.

```bash
python scripts/lambda_benchmark.py \
  --concurrency 1 4 8 \
  --region us-east-1
```

Default dataset: `benchmark/multi_model/alpaca_2k_prompts.jsonl`. The command deploys and warms nodes, marks the run `ready`, prints the dashboard command, and waits. It does **not** send measured Alpaca prompts automatically.

### Real E2E test (explicit opt-in)

```bash
RUN_LAMBDA_BENCHMARK_E2E=1 pytest tests/scripts/test_lambda_benchmark.py::test_real_lambda_benchmark_e2e -o addopts=
```

### Dashboard

```bash
streamlit run scripts/benchmark/dashboard.py
```

Sidebar defaults to `.lambda-runtime/benchmark.duckdb`. Filters: run ID, model, concurrency, prompt/output buckets, status. Charts: TTFT percentiles, TPOT histogram, throughput vs concurrency, E2E CDF, GPU memory/utilization, error bars, cost metric. Raw requests are in an expander drilldown only.

Prompt controls are shown above the charts once a run is `ready`:

- **Send Next Prompts** sends the next N dataset rows to every node and advances the persisted cursor.
- **Process Random Prompts** sends N randomly sampled rows to every node without advancing the cursor.
- **Prompts per model** and **Concurrency** are selected in the dashboard before sending.

## Model Matrix (MiniSGLang v1)

| Node | Model | HF Link | Notes |
|------|-------|---------|-------|
| 0 | Qwen/Qwen3-8B | [link](https://huggingface.co/Qwen/Qwen3-8B) | Dense Qwen3 decoder, Apache-2.0, ~8.2B params |
| 1 | Qwen/Qwen3-32B | [link](https://huggingface.co/Qwen/Qwen3-32B) | Dense Qwen3 decoder, Apache-2.0, 32.8B params |
| 2 | Qwen/Qwen3-30B-A3B | [link](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Qwen3 sparse MoE, Apache-2.0 |

Text-diffusion models are intentionally excluded (MiniSGLang serves causal LMs only).

## Known Limitations

1. **GPU metrics** require `nvidia-smi` over SSH; sampling is point-in-time, not continuous background polling.
2. **`/status`** does not expose GPU data; status samples capture idle-activity fields only.
3. **Streaming parser** assumes OpenAI-style SSE chunks; exotic chunk formats may under-count tokens.
4. **Hourly pricing** uses static lookup table, not live Lambda billing API.
5. **Concurrent sweep** runs per-node independently; cross-node load synchronization is not coordinated beyond identical configs.
6. **MoE model** (Qwen3-30B-A3B) may need substantial download time on first deploy; counted in model load, not request TTFT.
7. **E2E test** launches real billable instances; gated behind env flag by design.
8. **Dashboard batch execution** runs inside the Streamlit process, so the UI shows batch progress plus live OK/error and average TTFT/TPOT/E2E summaries while the batch runs; charts update after rerun. Request rows are inserted as individual requests complete.

## Review Fixes (Round 2)

Review against `plans/plan.md` identified five gaps before accepting the harness. All were addressed in this round.

### 1. Request payload shape

**Fix:** `DatasetRow.request_body()` and `warmup_request_body()` now send `ignore_eos=True` and `top_k=1` as top-level JSON fields on the chat completion payload, matching MiniSGLang's `OpenAICompletionRequest` schema (`python/minisgl/server/api_server.py`). Removed nesting under `extra_body`, which raw `urllib` requests do not expand.

**Rationale:** The benchmark harness posts JSON directly (not via the OpenAI SDK), so fields must appear at the top level for the server to honor deterministic sampling.

**Tests:** `test_dataset_request_body_uses_top_level_sampling_fields`

### 2. Actual concurrency

**Fix:** Blocking `urllib.request.urlopen` calls moved to `_stream_chat_completion_sync` and invoked from async code via `asyncio.to_thread`, so the semaphore in `run_benchmark_for_node` can release the event loop while requests run in parallel. Health/model polling and status sampling use the same pattern.

**Rationale:** Avoids adding `httpx`/`aiohttp` dependencies while fixing the prior behavior where async tasks blocked each other on I/O.

**Tests:** `test_concurrency_sweep_overlaps_requests` injects `_stream_chat_override` with an async delayed fake transport; asserts `max_in_flight >= 2` and wall time well below sequential execution.

### 3. DuckDB node upsert

**Fix:** Replaced `INSERT OR REPLACE` on `benchmark_nodes` with `INSERT ... ON CONFLICT DO UPDATE` using `COALESCE(excluded.col, benchmark_nodes.col)` for `model_load_s` and `warmup_s`. Deploy-time writes set `model_load_s`; warmup-time writes set only `warmup_s` without nulling deploy metadata.

**Rationale:** DuckDB `INSERT OR REPLACE` deletes and re-inserts the row, wiping columns omitted from the second insert.

**Tests:** `test_node_upsert_preserves_model_load_s_on_warmup_update`

### 4. Dashboard analytics

**Fix:**
- Filter logic allows rows with null `output_len_bucket` through the output-bucket sidebar filter (errors remain visible when status includes `error`).
- Throughput chart uses `request_throughput_per_s()` from `metrics.py`: request count divided by `(max(finished_at) - min(started_at))` per model/concurrency group, not mean per-request output tok/s.

**Rationale:** Error rows lack output token counts/buckets; throughput should reflect scheduler load (requests/sec), not decode speed.

**Tests:** `test_dashboard_filter_keeps_null_output_bucket_errors`, `test_request_throughput_uses_wall_clock`

### 5. Dry-run capacity selection

**Fix:** `run_dry_run()` accepts optional `select_capacity_fn` and `capacity_token` for dependency injection. CLI `--dry-run` defaults to `_mock_dry_run_capacity()` (first resolved instance type, no Lambda API). Tests inject real `select_instance_type_with_capacity` with monkeypatched capacity APIs. Dry-run now records selected `instance_type`, `region`, and hourly USD in run state and DuckDB.

**Rationale:** Plan requires exercising capacity-selection logic without launching instances; mock default keeps CLI usable offline while tests prove the real selector.

**Tests:** Updated `test_dry_run_mode` verifies H100 selection via injected selector and persisted DB metadata.

### Round 2 verification

```bash
.venv/bin/python3.13 -m pytest tests/scripts/test_lambda_benchmark.py -o addopts=
# 17 passed, 1 skipped

.venv/bin/python3.13 scripts/lambda_benchmark.py --dry-run
# succeeds without Lambda credentials
```

## Review Fixes (Round 3)

Follow-up review of Round 2 fixes found two remaining benchmark metric issues. Both are addressed below.

### 1. Dashboard request-throughput scaling

**Problem:** `_throughput_by_group()` converted `started_at` / `finished_at` with `astype("int64") / 1e9`. Pandas 3.0.3 returns microsecond-resolution integers for timestamp columns in this environment, so elapsed time was ~1000× too small and throughput ~1000× too high.

**Fix:** Compute wall-clock span with a timedelta: `(pd.to_datetime(wall_end) - pd.to_datetime(wall_start)).dt.total_seconds()`, then pass elapsed seconds to `request_throughput_per_s()`.

**Rationale:** Matches the cost-metric path in the same module (lines 193–195) and is unit-agnostic regardless of pandas datetime internal representation.

**Code:** `scripts/benchmark/dashboard.py` — `_throughput_by_group()`

**Tests:** `test_dashboard_throughput_uses_timedelta_wall_clock` — two requests spanning three seconds assert ~0.67 req/s, not hundreds.

### 2. Streamed output-token counting

**Problem:** `_stream_chat_completion_sync()` appended a timestamp for any SSE chunk with `choices`, including MiniSGLang's final `finish_reason: "stop"` chunk before `[DONE]`. That inflated `output_tokens` and TPOT.

**Fix:** Added `_stream_chunk_carries_content()`; only append timestamps when the chunk has non-empty `delta.content` and no `finish_reason`.

**Rationale:** Output token count should reflect decoded content chunks only; finish-only chunks carry no new tokens.

**Code:** `scripts/benchmark/runner.py` — `_stream_chunk_carries_content()`, `_stream_chat_completion_sync()`

**Tests:** `test_stream_completion_skips_finish_reason_stop_chunk` — two content chunks plus a stop chunk → `output_tokens == 2`.

### Round 3 verification

```bash
.venv/bin/python3.13 -m pytest tests/scripts/test_lambda_benchmark.py -o addopts=
# 19 passed, 1 skipped
```

## Future Work

- Periodic background GPU/status polling during benchmark execution
- Aggregate cross-model percentile views exported to JSON/Parquet
- Optional reuse of existing instances instead of always launching fresh
- Integrate `minisgl.benchmark.client` async OpenAI client for richer progress bars
- Pull live instance pricing from Lambda `/instance-types` when available
- Support configurable model matrix via CLI flags or YAML manifest
- Add request-level prompt/output token counts from server metadata when exposed

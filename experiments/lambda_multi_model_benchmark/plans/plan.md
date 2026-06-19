# Lambda Multi-Model Inference Benchmark Plan

## Summary

Create a reproducible Lambda Cloud benchmark that launches three same-type GPU
instances, serves one MiniSGLang-compatible Hugging Face model per instance,
warms each node, then lets the user send controlled prompt batches from the
Streamlit dashboard. Results are stored in DuckDB and visualized live as the
user sends batches. The launcher keeps SSH tunnels and Lambda instances alive
while the dashboard is in use, then terminates instances when the launcher is
interrupted.

Use MiniSGLang-only models for v1. Do not include text-diffusion models because
the repo currently serves causal LM architectures only.

Primary model matrix:

- `Qwen/Qwen3-8B`: dense Qwen3 decoder, Apache-2.0, 8.2B params.
- `mistralai/Mistral-7B-Instruct-v0.3`: dense Mistral decoder, Apache-2.0, 7B params.
- `Qwen/Qwen3-30B-A3B`: Qwen3 sparse MoE, Apache-2.0.

Sources to cite in the implementation notes:

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

## Key Implementation Changes

- Add a multi-node Lambda benchmark harness separate from the existing
  single-instance deploy flow.
  - Query Lambda `/instance-types`, prefer `gpu_1x_h100_sxm5`, then
    `gpu_1x_h100_pcie`, then existing A100 fallbacks.
  - Launch all three instances with the same instance type and region; if any
    launch or deploy step fails, terminate every instance from that run.
  - Store run state in ignored `.lambda-runtime/benchmark-runs/<run_id>.json`,
    not in the single `LAMBDA_PUBLIC_IP` `.env` field.

- Deploy the same repo archive and server config to each node.
  - Same Docker image/git SHA, same server args, same API key, same HF token,
    same `dtype=bfloat16`, `cache-type=radix`, `page-size=64`,
    `max-seq-len-override=4096`, and `cuda-graph-max-bs=64`.
  - Only `MINISGL_MODEL` differs per node.
  - Expose each remote server through SSH tunnels on local ports `19191`,
    `19192`, and `19193`.

- Standardize warmup and workload parity.
  - Wait for `/health` and `/v1/models`.
  - Run identical warmup prompts against each model and discard those
    measurements.
  - Use `mhenrichsen/alpaca_2k_test` converted to
    `benchmark/multi_model/alpaca_2k_prompts.jsonl` as the default workload.
    Keep the small sample JSONL for dry-run and local validation overrides.
  - Require a provided JSONL dataset with either `prompt` or `messages`, plus
    optional `id`, `max_tokens`, and metadata fields.
  - Do not automatically run measured dataset inference when nodes become
    ready. The dashboard controls prompt count, random-vs-next selection, and
    concurrency, and sends the selected prompts to every warm model node.

- Persist benchmark data in DuckDB.
  - Add dependencies: `duckdb`, `streamlit`, `plotly`, `pandas`, and `pyarrow`.
  - Store run metadata, model/node metadata, request-level timings, per-token
    stream timestamps, status samples, GPU samples, errors, and cleanup results.
  - Compute TTFT, inter-token latency/TPOT, end-to-end latency, output
    tokens/sec, request throughput, p50/p90/p95/p99, error rate, GPU
    memory/utilization, model load time, and estimated cost per 1M tokens.

- Add a Streamlit dashboard.
  - Prompt controls: "send next N prompts" and "process N random prompts",
    where N is prompts per model and concurrency is selected by the user.
  - Filters: run ID, model, concurrency, prompt length bucket, output length
    bucket, status.
  - Charts: TTFT percentiles by model, TPOT distribution,
    throughput-vs-concurrency curve, E2E latency CDF, GPU memory/utilization
    over time, error-rate bars, and cost per 1M output tokens.
  - Prefer charts over raw tables; keep raw request table available only as a
    drilldown.

- Add mandatory cleanup.
  - Wrap orchestration in `try/finally`.
  - Handle SIGINT/SIGTERM by stopping local tunnels, terminating all launched
    Lambda instance IDs, and writing cleanup status to DuckDB.
  - After termination, poll Lambda `/instances` and fail loudly if any
    benchmark-named instance remains active or booting.

## Test Plan

- Unit test Lambda instance selection with mocked `/instance-types`, including
  H100 capacity, fallback capacity, and no-capacity failure.
- Unit test cleanup behavior so partial launch, deploy failure, benchmark
  failure, and keyboard interrupt all terminate tracked instance IDs.
- Unit test DuckDB schema creation and inserts using a temporary database.
- Unit test dataset parsing for `prompt`, `messages`, missing `max_tokens`,
  malformed JSONL, and stable dataset hash.
- Unit test benchmark metric calculations from synthetic streaming timestamps.
- Add a dry-run mode that exercises capacity selection, run-state creation,
  model matrix validation, and dashboard DB initialization without launching
  Lambda nodes.
- Keep real Lambda E2E tests behind an explicit environment flag such as
  `RUN_LAMBDA_BENCHMARK_E2E=1`.
  The E2E path launches, deploys, warms nodes, and then returns through an
  injected wait hook without sending measured Alpaca prompts automatically.

## Review Fix Plan

Review against this plan and `plans/implementation-spec.md` found the
following fixes are needed before accepting the benchmark harness:

- Fix raw benchmark request payloads so `ignore_eos=True` and `top_k=1` are
  sent as top-level MiniSGLang fields, not nested under `extra_body`; add a
  unit test for `DatasetRow.request_body()` and warmup request body shape.
- Make the concurrency sweep actually concurrent by replacing blocking
  `urllib` calls inside async tasks with an async client/transport, or by
  isolating blocking calls in worker threads; add a delayed fake-transport test
  proving concurrency greater than 1 overlaps requests.
- Preserve `benchmark_nodes.model_load_s` when recording warmup time; replace
  node `INSERT OR REPLACE` behavior with an update/upsert that does not null
  existing node metadata, and add a regression test for deploy-time then
  warmup-time writes.
- Fix dashboard analytics so error rows with null `output_len_bucket` remain
  visible under the status filter, and compute request throughput as request
  count divided by elapsed wall time for each model/concurrency group rather
  than mean output tokens/sec.
- Bring dry-run behavior back in line with this plan by exercising mocked or
  dependency-injected capacity-selection logic without launching instances, or
  update the implementation spec if dry-run intentionally avoids Lambda API
  access.

## Review Verification

- Installed the missing benchmark runtime packages into the existing workspace
  `.venv`: `duckdb`, `streamlit`, `plotly`, `pandas`, and `pyarrow`.
- Verified the focused benchmark tests with
  `.venv/bin/python -m pytest tests/scripts/test_lambda_benchmark.py -o addopts=`;
  result: 12 passed, 1 skipped.
- Verified dry-run with `.venv/bin/python scripts/lambda_benchmark.py --dry-run`;
  result: run state and DuckDB database were created under `.lambda-runtime/`.

## Follow-Up Review Fix Plan

Review of the Round 2 fixes found two remaining benchmark metric issues:

- Fix dashboard request-throughput scaling. `_throughput_by_group()` currently
  converts `started_at` and `finished_at` timestamps with
  `astype("int64") / 1e9`, but pandas 3.0.3 returns microsecond-resolution
  integers for these timestamp columns in this environment. This reports
  throughput roughly 1000x too high. Compute elapsed wall time with a
  timedelta, such as `(max_finished - min_started).total_seconds()`, or
  otherwise normalize based on the actual timestamp unit. Add a regression test
  where two requests spanning three seconds report about `0.67 req/s`, not
  hundreds of requests per second.
- Fix streamed output-token counting. `_stream_chat_completion_sync()` appends a
  timestamp for any SSE chunk with `choices`, including the final
  `finish_reason: "stop"` chunk emitted by MiniSGLang before `[DONE]`. That
  inflates output token count and TPOT. Only append timestamps for chunks that
  carry actual generated content, or skip chunks with non-null `finish_reason`
  / empty delta. Add a regression test with two content chunks plus a final
  stop chunk and assert only two output chunks are counted.

## Assumptions And Defaults

- "Same node" means same Lambda instance type and region, not same physical
  host.
- DuckDB is the local analytics store; SQLite is not used for this benchmark.
- Streamlit is the dashboard framework.
- Text diffusion is out of scope for v1 because MiniSGLang currently supports
  causal LM architectures in this repo.
- The benchmark excludes cold-start and model-download time from request latency
  metrics, but stores model load and warmup duration separately.
- Real runs intentionally keep Lambda nodes and SSH tunnels alive after warmup
  so the user can drive prompt batches from the dashboard. Interrupting the
  launcher remains the cleanup path and terminates tracked instances.

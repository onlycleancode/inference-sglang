# Lambda Multi-Model Benchmark Experiment

## Goal

This experiment evaluates MiniSGLang inference behavior across three same-type
Lambda Cloud GPU instances. Each node serves one MiniSGLang-compatible
Hugging Face model. The launcher deploys and warms the nodes, then stops at a
dashboard-ready state without sending measured prompts automatically. The user
controls prompt batches from Streamlit and watches request latency, streaming
token timing, node metadata, cleanup status, and analytics accumulate in
DuckDB.

## Contents

- `scripts/lambda_benchmark.py` - CLI entrypoint for dry-run and real Lambda
  benchmark orchestration.
- `scripts/benchmark/` - experiment-only benchmark package: configuration,
  Lambda instance launch helpers, deployment, cleanup, dataset parsing,
  metrics, DuckDB persistence, runner, dashboard batch control, and Streamlit
  dashboard.
- `benchmark/multi_model/alpaca_2k_prompts.jsonl` - default 2,000-prompt
  workload converted from `mhenrichsen/alpaca_2k_test`.
- `benchmark/multi_model/sample_dataset.jsonl` - sample workload used for
  dry-run and local validation.
- `tests/scripts/test_lambda_benchmark.py` - focused unit tests and gated
  Lambda E2E test for this benchmark harness.
- `plans/plan.md` - original implementation plan plus review notes.
- `plans/implementation-spec.md` - implementation decisions, verification
  notes, and known limitations.
- `STARTUP_RUNBOOK.md` - operational startup guide from the first successful
  real three-node run, including preflight checks, known failure modes, and
  validation commands.

## Core Repo Integration

The experiment depends on existing core repo behavior but does not include core
MiniSGLang functionality in this folder. The only integration points outside
this folder are:

- `pyproject.toml` optional `benchmark` dependencies:
  `duckdb`, `streamlit`, `plotly`, `pandas`, and `pyarrow`.
- `.gitignore` entries for local runtime/plan artifacts.
- Existing Lambda helpers under `scripts/lambda_common.py`,
  `scripts/lambda_deploy.py`, and `scripts/package_lambda.sh`.
- Existing MiniSGLang OpenAI-compatible server endpoints.

## Verification

Latest focused verification:

```bash
.venv/bin/python -m pytest tests/scripts/test_lambda_benchmark.py -o addopts=
# 22 passed, 1 skipped
```

```bash
.venv/bin/python3.13 -m pytest tests/scripts/test_lambda_benchmark.py -o addopts=
# 22 passed, 1 skipped
```

```bash
.venv/bin/python scripts/lambda_benchmark.py --dry-run
# creates run state and DuckDB under .lambda-runtime/
```

Real deployment defaults to the Alpaca 2k workload:

```bash
.venv/bin/python scripts/lambda_benchmark.py --concurrency 1 4 8
```

After deployment and warmup, open the printed Streamlit dashboard command. Use
the dashboard controls to send the next N prompts or N random prompts; N is per
model node. Press Ctrl-C in the launcher when finished to terminate the
instances and tunnels.

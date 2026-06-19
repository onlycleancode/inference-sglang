# Lambda Multi-Model Startup Runbook

This runbook records the operational lessons from the first successful
three-model Lambda startup on 2026-06-19. Use it before the next run so startup
does not rediscover the same launch, dependency, and readiness failures.

## Why Earlier Runs Kept Terminating

The benchmark launcher owns the paid Lambda instances and SSH tunnels. Its
cleanup manager intentionally terminates all tracked benchmark instances when
the launcher exits, receives Ctrl-C, or raises during deploy/warmup. That is the
right default because leaked H100 instances are expensive.

Most restarts happened because the current run had already booted VMs with an
archive or Docker image that was missing a required fix. Once the package,
Dockerfile, runtime environment, or launcher logic changed, the safe path was to
let cleanup terminate the bad run and launch a fresh cluster with one coherent
archive and run state. Manually patching live nodes would have bypassed the
run-state JSON, DuckDB node metadata, and tunnel ownership model.

Specific failures fixed during the successful run:

- Lambda rejected `quantity=3` launch requests. The harness now launches three
  same-type instances one at a time.
- Remote deploy crashed locally because `subprocess.run(..., text=True)` was
  given bytes input. The SSH script is now passed as text.
- Mistral startup needed tokenizer dependencies. `protobuf` and `sentencepiece`
  are now base project dependencies.
- Qwen3 MoE needed Python headers for runtime Triton compilation.
  `python${PYTHON_VERSION}-dev` is now installed in the runtime Docker stage.
- Runtime codegen could not infer H100 compute capability. Deployment now sets
  `TVM_FFI_CUDA_ARCH_LIST=9.0` for H100 instance types and `8.0` for A100
  fallbacks.
- Warmup was stuck polling `/v1/models` without auth. Readiness checks now send
  `Authorization: Bearer $MINISGL_API_KEY`.
- Local dashboard and validation scripts may need permission to bind/connect to
  local ports in the Codex sandbox. Use approved elevation for Streamlit and
  localhost tunnel validation when needed.

## Canonical Commands

Run the benchmark from the repository root:

```bash
cd /Users/joelmontano/Documents/Projects/mini-sglang
.venv/bin/python scripts/lambda_benchmark.py --concurrency 1 4 8
```

The files under this experiment directory are a preserved experiment snapshot.
The active run path is the top-level harness under `scripts/benchmark/` plus
the root `Dockerfile` and `pyproject.toml`. If you intentionally run the
experiment-local snapshot, synchronize it with the top-level fixes first.

## Preflight Checklist

Run these checks before provisioning:

```bash
.venv/bin/python -m pytest tests/scripts/test_lambda_benchmark.py -o addopts=
```

Expected after the 2026-06-19 fixes:

```text
26 passed, 1 skipped
```

Confirm local credentials exist without printing secret values:

```bash
.venv/bin/python -c "import os, sys; from pathlib import Path; sys.path.insert(0, 'scripts'); from lambda_common import ENV_FILE, load_env_file; load_env_file(ENV_FILE); print({k: bool(os.getenv(k)) for k in ['LAMBDA_CLOUD_API_KEY', 'MINISGL_API_KEY', 'HF_TOKEN']}); p=Path(os.getenv('SSH_PRIVATE_KEY_PATH', str(Path.home()/'.ssh/id_ed25519'))); print(p, p.exists(), Path(str(p)+'.pub').exists())"
```

Confirm there are no active benchmark instances before launching:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from lambda_common import ENV_FILE, load_env_file, require, list_instances; load_env_file(ENV_FILE); token=require('LAMBDA_CLOUD_API_KEY'); print([(i.get('id'), i.get('name'), i.get('status'), i.get('ip')) for i in list_instances(token) if 'minisgl-benchmark' in (i.get('name') or '')])"
```

Rows in `terminating` state can linger in the Lambda API after cleanup. Do not
treat them as usable nodes. The cleanup invariant is no matching benchmark
instances in `active` or `booting` state from an old run.

Check local ports are free:

```bash
lsof -nP -iTCP:19191 -iTCP:19192 -iTCP:19193 -iTCP:8501 -sTCP:LISTEN
```

## Startup Flow

1. Start the launcher from the repo root:

   ```bash
   .venv/bin/python scripts/lambda_benchmark.py --concurrency 1 4 8
   ```

2. Keep the launcher process running. It owns the SSH tunnels and is also the
   cleanup path. Pressing Ctrl-C terminates the benchmark instances.

3. Expect these stages:

   ```text
   Selected gpu_1x_h100_sxm5 in <region>
   package_lambda.sh creates dist/minisgl-lambda-*.tar.gz
   scp archive to node 0, deploy Qwen/Qwen3-8B
   scp archive to node 1, deploy mistralai/Mistral-7B-Instruct-v0.3
   scp archive to node 2, deploy Qwen/Qwen3-30B-A3B
   warmup through local tunnels 19191, 19192, 19193
   Nodes are warm and ready...
   ```

4. Deployment is sequential. A full build and model startup can take several
   minutes per node. Avoid interrupting a quiet launcher unless logs prove it is
   unrecoverable.

5. The run-state file is authoritative for the current run:

   ```bash
   ls -t .lambda-runtime/benchmark-runs/*.json | head -1
   ```

   A ready run has:

   ```json
   "status": "ready"
   ```

   and each node has `ip` plus `tunnel_pid`.

## Health Checks

Remote node health:

```bash
ssh -i "$SSH_PRIVATE_KEY_PATH" ubuntu@<node-ip> \
  'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:1919/health'
```

Local tunnel health after the launcher starts tunnels:

```bash
curl --max-time 10 -s -o /dev/null -w '19191 %{http_code}\n' http://127.0.0.1:19191/health
curl --max-time 10 -s -o /dev/null -w '19192 %{http_code}\n' http://127.0.0.1:19192/health
curl --max-time 10 -s -o /dev/null -w '19193 %{http_code}\n' http://127.0.0.1:19193/health
```

All should return `200`.

If a node is not healthy, read logs:

```bash
ssh -i "$SSH_PRIVATE_KEY_PATH" ubuntu@<node-ip> \
  'sudo docker ps -a --format "{{.Names}} {{.Status}} {{.Ports}}"; sudo docker logs lambda-minisgl-1 --tail 80 2>&1'
```

## Dashboard

Start Streamlit after the run is ready:

```bash
.venv/bin/streamlit run scripts/benchmark/dashboard.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

Dashboard URL:

```text
http://127.0.0.1:8501
```

In the Codex sandbox this may require elevated permission because Streamlit
binds a local server socket.

## Five-Prompt Validation

After the launcher marks the run ready, send five measured prompts through the
same control path used by the dashboard. Replace `RUN_ID` with the current
ready run ID:

```bash
.venv/bin/python - <<'PY'
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
from lambda_common import ENV_FILE, load_env_file, require
from benchmark.config import BenchmarkConfig, DEFAULT_DB_PATH
from benchmark.control import NodeTarget, run_batch_for_nodes, select_next_rows
from benchmark.dataset import load_dataset
from benchmark.duckdb_store import BenchmarkStore

run_id = 'RUN_ID'
load_env_file(ENV_FILE)
api_key = require('MINISGL_API_KEY')
config = BenchmarkConfig()
dataset = load_dataset(config.dataset_path)
store = BenchmarkStore(DEFAULT_DB_PATH)
try:
    nodes_rows = store.conn.execute(
        """
        SELECT node_index, model_id, local_port, ip
        FROM benchmark_nodes
        WHERE run_id = ?
        ORDER BY node_index
        """,
        [run_id],
    ).fetchall()
    nodes = [NodeTarget(int(i), str(model), int(port), ip) for i, model, port, ip in nodes_rows]
    cursor = store.get_next_row_index(run_id)
    rows, next_cursor = select_next_rows(dataset, cursor, 5)

    def progress(request_id, node_index, row_index, status):
        print(f'node={node_index} row={row_index} status={status} request={request_id}', flush=True)

    asyncio.run(
        run_batch_for_nodes(
            config=config,
            store=store,
            run_id=run_id,
            nodes=nodes,
            api_key=api_key,
            rows=rows,
            concurrency=1,
            progress=progress,
            ssh_key=Path(os.getenv('SSH_PRIVATE_KEY_PATH', str(Path.home() / '.ssh/id_ed25519'))),
        )
    )
    store.set_next_row_index(run_id, next_cursor)
finally:
    store.close()
PY
```

If this runs inside a sandboxed environment, it may need permission to connect
to the local SSH tunnel sockets.

Verify DuckDB collection:

```bash
.venv/bin/python - <<'PY'
import duckdb
run_id = 'RUN_ID'
conn = duckdb.connect('.lambda-runtime/benchmark.duckdb', read_only=True)
print(conn.execute('select run_id, status from benchmark_runs where run_id=?', [run_id]).fetchall())
print(conn.execute('''
    select node_index, model_id, status, count(*), sum(output_tokens)
    from benchmark_requests
    where run_id=? and is_warmup=false
    group by node_index, model_id, status
    order by node_index
''', [run_id]).fetchall())
print(conn.execute('''
    select count(*)
    from benchmark_stream_tokens
    where request_id in (
        select request_id from benchmark_requests where run_id=?
    )
''', [run_id]).fetchone())
conn.close()
PY
```

Expected validation result:

```text
node 0: ok, 5 rows
node 1: ok, 5 rows
node 2: ok, 5 rows
stream token rows > 0
```

If the validation rows show `Operation not permitted`, those rows came from a
local sandbox denial, not the remote models. Delete those invalid rows, reset
the control cursor if needed, and rerun validation with local network/tunnel
permission.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Lambda launch says `Requested too many instances` | Account/region only allows `quantity=1` launches | Use the fixed harness that launches nodes individually |
| Local deploy crashes with `bytes object has no attribute encode` | SSH script passed as bytes while `text=True` | Use the fixed `deploy.py` text input path |
| Mistral fails with missing `sentencepiece` or `protobuf` | Tokenizer dependencies absent from image | Ensure root `pyproject.toml` includes `sentencepiece` and `protobuf` |
| Qwen3 MoE fails with `Python.h: No such file or directory` | Triton compiles at runtime and needs Python headers | Ensure runtime Docker stage installs `python${PYTHON_VERSION}-dev` |
| Qwen3 MoE fails with `Could not detect CUDA compute_cap` | `tvm_ffi` cannot query compute cap inside container | Ensure deploy writes `TVM_FFI_CUDA_ARCH_LIST=9.0` for H100 |
| Warmup loops with repeated `/v1/models` 401 logs | Readiness probe lacks API key | Ensure `wait_for_models(..., api_key=api_key)` is used |
| DuckDB reports file lock during launch | Launcher owns a write connection | Wait until launcher reaches ready; it closes the write connection before dashboard use |
| Dashboard cannot bind `127.0.0.1:8501` | Local sandbox denied server bind | Start Streamlit with approved local bind permission |
| Prompt validation inserts 15 error rows with `Operation not permitted` | Local sandbox denied Python localhost tunnel connections | Remove those rows and rerun validation with approved local network permission |

## Cleanup

When testing is finished, press Ctrl-C in the launcher process. Do not kill the
SSH tunnel PIDs directly unless the launcher is already gone. The launcher
cleanup path stops tunnels, terminates tracked Lambda instance IDs, records
cleanup telemetry in DuckDB, and checks for remaining active/booting benchmark
instances.

Verify cleanup:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from lambda_common import ENV_FILE, load_env_file, require, list_instances; load_env_file(ENV_FILE); token=require('LAMBDA_CLOUD_API_KEY'); print([(i.get('id'), i.get('name'), i.get('status')) for i in list_instances(token) if 'minisgl-benchmark' in (i.get('name') or '')])"
```

`terminating` rows can appear briefly after cleanup. There should be no old
benchmark rows in `active` or `booting` state.

## Known-Good Evidence From 2026-06-19

Successful run:

```text
run_id: 20260619T182521Z-a6dcb537
instance_type: gpu_1x_h100_sxm5
region: us-south-2
models:
  node 0: Qwen/Qwen3-8B, local port 19191
  node 1: mistralai/Mistral-7B-Instruct-v0.3, local port 19192
  node 2: Qwen/Qwen3-30B-A3B, local port 19193
validation:
  5 measured ok rows per node
  6 warmup rows
  stream-token rows recorded
dashboard:
  http://127.0.0.1:8501
```

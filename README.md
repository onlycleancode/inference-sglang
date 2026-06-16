# Learning Inference!!!

This is a personal learning repo forked from
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang).
It keeps the core Mini-SGLang implementation and adds a small Lambda Cloud
deployment workflow for serving Qwen.

This is not an official SGLang project. The goal is to understand the moving
parts of an LLM inference server while keeping the repository small enough to
read, modify, and deploy.

## What Is Included

- Core Mini-SGLang source under `python/minisgl/`
- Upstream-style tests and benchmarks under `tests/` and `benchmark/`
- A Dockerfile for GPU serving
- Lambda Cloud deployment assets under `deploy/lambda/`
- Local deployment, tunnel, watchdog, chat UI, and smoke-test scripts under
  `scripts/`

Local notes, retrospectives, generated archives, runtime logs, private
environment files, virtual environments, and cache directories are intentionally
ignored by Git.

## Security Model

If you wanna for you'll need:

- `LAMBDA_CLOUD_API_KEY`
- `HF_TOKEN`
- `MINISGL_API_KEY`
- `SSH_PRIVATE_KEY_PATH`
- `LAMBDA_PUBLIC_IP` after an instance is launched

The Lambda deployment binds Mini-SGLang to localhost on the remote VM and uses
an SSH tunnel for access from your machine. Public inference ports are not
required. Deployed API routes require `Authorization: Bearer $MINISGL_API_KEY`.

## Requirements

For local development:

- Linux with an NVIDIA GPU and compatible CUDA driver
- Python 3.10+
- `uv` or another Python environment manager

For Lambda serving:

- Lambda Cloud API key
- SSH public key uploaded to Lambda Cloud
- Hugging Face token with access to the model you want to run
- Docker and NVIDIA runtime support on the Lambda VM

The default main model is `Qwen/Qwen3-8B` but more support will be added as I learn more about inference :)

## Local Install

```bash
git clone <your-fork-url>
cd mini-sglang

uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
```

Run a small local server:

```bash
export HF_TOKEN=<your-hugging-face-token>
export MINISGL_API_KEY=<long-random-api-key>
export MINISGL_REQUIRE_API_KEY=1

python -m minisgl \
  --model Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 1919
```

Call it with an OpenAI-compatible request:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${MINISGL_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain KV cache reuse in one paragraph."}],
    "max_tokens": 128
  }'
```

## Lambda Deployment

Create a root `.env` file. It is ignored by Git.

```dotenv
LAMBDA_CLOUD_API_KEY=<your-lambda-api-key>
LAMBDA_REGION=us-east-1
LAMBDA_INSTANCE_TYPE=gpu_1x_h100_sxm5
LAMBDA_INSTANCE_FALLBACK=gpu_1x_a100_sxm4,gpu_1x_a100

SSH_PRIVATE_KEY_PATH=/path/to/your/lambda/private/key

MINISGL_MODEL=Qwen/Qwen3-8B
MINISGL_SMOKE_MODEL=Qwen/Qwen3-0.6B
MINISGL_PORT=1919
MINISGL_API_KEY=<long-random-api-key>
MINISGL_REQUIRE_API_KEY=1

HF_TOKEN=<your-hugging-face-token>
MINISGL_IDLE_TIMEOUT_S=3600
```

Make the shell scripts executable:

```bash
chmod +x scripts/*.sh
```

Deploy the smoke model first:

```bash
./scripts/lambda_turn_on.sh --smoke --no-browser
```

When the smoke model works, redeploy or reuse the instance for the main Qwen
model:

```bash
./scripts/lambda_turn_on.sh --reuse --no-browser
```

The turn-on script:

- provisions or reuses a Lambda GPU instance
- packages this repo without secrets
- copies and deploys the package on the VM
- starts Docker Compose on the VM
- opens a local SSH tunnel to `127.0.0.1:1919`
- starts an idle watchdog
- serves the local chat UI at `http://127.0.0.1:8765/`

Run a request through the tunnel:

```bash
python scripts/minisgl_remote_infer.py \
  "Explain the difference between prefill and decode in LLM serving."
```

Turn everything off when finished:

```bash
python scripts/lambda_turn_off.py
```

Use `--keep-instance` if you want to stop the server but leave the Lambda VM
running.

## Manual Package Path

You can build the deployment tarball without launching a Lambda instance:

```bash
./scripts/package_lambda.sh
```

The archive is written to `dist/`, which is ignored. It contains source,
deployment assets, and scripts, but excludes local `.env` files and Python
caches.

On a manually created Lambda VM:

```bash
mkdir -p ~/minisgl
tar -xzf minisgl-lambda-*.tar.gz -C ~/minisgl
cd ~/minisgl/deploy/lambda
cp .env.example .env
# Fill MINISGL_API_KEY and HF_TOKEN in .env.
docker compose up -d --build
```

From your local machine:

```bash
ssh -N -L 1919:127.0.0.1:1919 ubuntu@<lambda-public-ip>
python scripts/minisgl_remote_infer.py "Say hello from Qwen."
```

## Useful Commands

```bash
# Check what would go into a package.
./scripts/package_lambda.sh --dry-run

# Start only the SSH tunnel when the VM is already running.
./scripts/lambda_tunnel.sh

# Run a health/smoke check through the local tunnel.
python scripts/minisgl_smoke_check.py

# Serve the local chat UI without provisioning.
python scripts/serve_lambda_chat.py
```

## Before Pushing

Run these checks before publishing:

```bash
git status --short --ignored
git add -n .
```

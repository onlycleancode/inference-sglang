"""Benchmark configuration and model matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".lambda-runtime"
BENCHMARK_RUNS_DIR = RUNTIME_DIR / "benchmark-runs"
DEFAULT_DB_PATH = RUNTIME_DIR / "benchmark.duckdb"
SAMPLE_DATASET = ROOT / "benchmark" / "multi_model" / "sample_dataset.jsonl"
ALPACA_2K_DATASET = ROOT / "benchmark" / "multi_model" / "alpaca_2k_prompts.jsonl"

# Local tunnel ports map 1:1 to model nodes (remote port stays 1919).
TUNNEL_PORTS = (19191, 19192, 19193)
REMOTE_PORT = 1919
NODE_COUNT = 3

BENCHMARK_INSTANCE_PREFIX = "minisgl-benchmark"

# MiniSGLang-only model matrix for v1.
MODEL_MATRIX: tuple[tuple[str, int], ...] = (
    ("Qwen/Qwen3-8B", TUNNEL_PORTS[0]),
    ("mistralai/Mistral-7B-Instruct-v0.3", TUNNEL_PORTS[1]),
    ("Qwen/Qwen3-30B-A3B", TUNNEL_PORTS[2]),
)

# Standardized server args for deploy parity across nodes.
SERVER_ARGS: tuple[str, ...] = (
    "--dtype",
    "bfloat16",
    "--cache-type",
    "radix",
    "--page-size",
    "64",
    "--max-seq-len-override",
    "4096",
    "--cuda-graph-max-bs",
    "64",
)

DEFAULT_WARMUP_PROMPTS = (
    "Warmup: summarize the concept of attention in one sentence.",
    "Warmup: list three primary colors.",
)

# Approximate USD/hour when Lambda API does not expose pricing (H100/A100 ballpark).
INSTANCE_HOURLY_USD: dict[str, float] = {
    "gpu_1x_h100_sxm5": 3.29,
    "gpu_1x_h100_pcie": 2.99,
    "gpu_1x_a100_sxm4": 1.79,
    "gpu_1x_a100": 1.29,
}


@dataclass
class BenchmarkConfig:
    """Runtime configuration for a benchmark run."""

    dataset_path: Path = ALPACA_2K_DATASET
    db_path: Path = DEFAULT_DB_PATH
    region: str = "us-east-1"
    concurrency_levels: tuple[int, ...] = (1, 4, 8)
    default_max_tokens: int = 128
    warmup_prompts: tuple[str, ...] = DEFAULT_WARMUP_PROMPTS
    dry_run: bool = False
    run_id: str | None = None
    models: tuple[tuple[str, int], ...] = field(default_factory=lambda: MODEL_MATRIX)

    def validate(self) -> None:
        if len(self.models) != NODE_COUNT:
            raise ValueError(f"Expected {NODE_COUNT} models, got {len(self.models)}")
        ports = [port for _, port in self.models]
        if len(set(ports)) != len(ports):
            raise ValueError("Model tunnel ports must be unique")
        if not self.dataset_path.exists() and not self.dry_run:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

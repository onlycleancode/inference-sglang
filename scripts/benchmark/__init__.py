"""Multi-model Lambda inference benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

__all__ = [
    "MODEL_MATRIX",
    "BenchmarkConfig",
    "BenchmarkCleanupManager",
    "BenchmarkStore",
    "compute_request_metrics",
    "load_dataset",
    "select_instance_type_with_capacity",
]

from benchmark.cleanup import BenchmarkCleanupManager
from benchmark.config import BenchmarkConfig, MODEL_MATRIX
from benchmark.dataset import load_dataset
from benchmark.duckdb_store import BenchmarkStore
from benchmark.instance import select_instance_type_with_capacity
from benchmark.metrics import compute_request_metrics

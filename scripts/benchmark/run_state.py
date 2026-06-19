"""Persist benchmark run state under .lambda-runtime/benchmark-runs/."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.config import BENCHMARK_RUNS_DIR


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass
class NodeState:
    node_index: int
    model_id: str
    local_port: int
    instance_id: str | None = None
    ip: str | None = None
    tunnel_pid: int | None = None


@dataclass
class RunState:
    run_id: str
    created_at: str
    status: str = "created"
    instance_type: str | None = None
    region: str | None = None
    dataset_path: str | None = None
    dataset_hash: str | None = None
    git_sha: str | None = None
    archive_path: str | None = None
    nodes: list[NodeState] = field(default_factory=list)
    instance_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None = None,
        models: list[tuple[str, int]],
    ) -> RunState:
        rid = run_id or new_run_id()
        nodes = [
            NodeState(node_index=i, model_id=model_id, local_port=port)
            for i, (model_id, port) in enumerate(models)
        ]
        return cls(
            run_id=rid,
            created_at=datetime.now(timezone.utc).isoformat(),
            nodes=nodes,
        )

    def path(self) -> Path:
        BENCHMARK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        return BENCHMARK_RUNS_DIR / f"{self.run_id}.json"

    def save(self) -> Path:
        path = self.path()
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, run_id: str) -> RunState:
        path = BENCHMARK_RUNS_DIR / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text())
        nodes = [NodeState(**node) for node in raw.pop("nodes", [])]
        return cls(nodes=nodes, **raw)

#!/usr/bin/env python3
"""Launch a three-node Lambda multi-model MiniSGLang benchmark."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lambda_common import ENV_FILE, load_env_file, require  # noqa: E402

from benchmark.cleanup import BenchmarkCleanupManager, start_ssh_tunnel  # noqa: E402
from benchmark.config import (  # noqa: E402
    BenchmarkConfig,
    CUDA_ARCH_BY_INSTANCE_TYPE,
    INSTANCE_HOURLY_USD,
    MODEL_MATRIX,
)
from benchmark.deploy import git_sha, package_repo, remote_deploy_node, wait_for_ssh  # noqa: E402
from benchmark.duckdb_store import BenchmarkStore  # noqa: E402
from benchmark.instance import (  # noqa: E402
    SelectedCapacity,
    launch_benchmark_cluster,
    select_instance_type_with_capacity,
    wait_for_instance_ips,
)
from benchmark.models import validate_model_matrix  # noqa: E402
from benchmark.run_state import NodeState, RunState, new_run_id  # noqa: E402
from benchmark.runner import prepare_nodes_for_dashboard  # noqa: E402

WaitForControlFn = Callable[[str, Path], None]


@dataclass(frozen=True)
class NodeDeployResult:
    node_index: int
    model_load_s: float


def _deploy_one_node(
    *,
    node: NodeState,
    ip: str,
    ssh_key: Path,
    archive: Path,
    api_key: str,
    hf_token: str,
    cuda_arch_list: str | None,
) -> NodeDeployResult:
    wait_for_ssh(ip, ssh_key)
    model_load_s = remote_deploy_node(
        ip=ip,
        ssh_key=ssh_key,
        archive=archive,
        model_id=node.model_id,
        api_key=api_key,
        hf_token=hf_token,
        cuda_arch_list=cuda_arch_list,
    )
    return NodeDeployResult(node_index=node.node_index, model_load_s=model_load_s)


def _deploy_nodes_parallel(
    *,
    nodes: list[NodeState],
    ips: dict[str, str],
    ssh_key: Path,
    archive: Path,
    api_key: str,
    hf_token: str,
    cuda_arch_list: str | None,
) -> dict[int, float]:
    """Deploy all benchmark nodes concurrently and return model load seconds."""
    deploy_targets: list[tuple[NodeState, str]] = []
    for node in nodes:
        if node.instance_id is None:
            raise RuntimeError(f"Node {node.node_index} is missing an instance ID")
        ip = ips[node.instance_id]
        node.ip = ip
        deploy_targets.append((node, ip))

    if not deploy_targets:
        return {}

    print(f"Deploying {len(deploy_targets)} benchmark nodes in parallel...")
    results: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=len(deploy_targets)) as executor:
        futures = {
            executor.submit(
                _deploy_one_node,
                node=node,
                ip=ip,
                ssh_key=ssh_key,
                archive=archive,
                api_key=api_key,
                hf_token=hf_token,
                cuda_arch_list=cuda_arch_list,
            ): node.node_index
            for node, ip in deploy_targets
        }
        for future in as_completed(futures):
            result = future.result()
            results[result.node_index] = result.model_load_s
            print(f"Node {result.node_index} deployed in {result.model_load_s:.1f}s")
    return results


def _find_ssh_key_name(token: str, public_key: str) -> str:
    from lambda_common import api_request

    keys = api_request("GET", "/ssh-keys", token).get("data", [])
    for item in keys:
        if item.get("public_key", "").strip() == public_key:
            return item["name"]
    raise SystemExit("SSH public key is not registered in Lambda.")


def _read_public_key(private_key_path: Path) -> str:
    pub_path = Path(str(private_key_path) + ".pub")
    if not pub_path.exists():
        raise SystemExit(f"Missing public key: {pub_path}")
    return pub_path.read_text().strip()


def _mock_dry_run_capacity(
    _token: str,
    *,
    preferred_region: str | None = None,
    instance_types: list[str] | None = None,
) -> SelectedCapacity:
    """Default dry-run capacity pick (no Lambda API); prefers first resolved type."""
    from lambda_common import resolve_instance_types

    ordered = instance_types or resolve_instance_types()
    return SelectedCapacity(instance_type=ordered[0], region=preferred_region or "us-east-1")


def _archive_existing_dashboard_db(db_path: Path, *, run_id: str) -> Path | None:
    """Move the previous dashboard DB aside so each launcher run starts clean."""
    if not db_path.parent.exists():
        return None

    db_files = sorted(path for path in db_path.parent.glob(f"{db_path.name}*") if path.is_file())
    if not db_files:
        return None

    archive_parent = db_path.parent / "benchmark-db-archive"
    archive_parent.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_parent / run_id
    suffix = 2
    while archive_dir.exists():
        archive_dir = archive_parent / f"{run_id}-{suffix}"
        suffix += 1
    archive_dir.mkdir()

    for db_file in db_files:
        db_file.rename(archive_dir / db_file.name)
    print(f"Archived previous dashboard DB files to: {archive_dir}")
    return archive_dir


def run_dry_run(
    config: BenchmarkConfig,
    *,
    select_capacity_fn=None,
    capacity_token: str = "dry-run",
) -> int:
    """Exercise selection, run-state, model matrix, and DuckDB init without Lambda."""
    from collections.abc import Callable

    from benchmark.dataset import load_dataset
    from benchmark.instance import SelectedCapacity

    validate_model_matrix(config.models)

    select_fn: Callable[..., SelectedCapacity] = select_capacity_fn or _mock_dry_run_capacity
    capacity = select_fn(capacity_token, preferred_region=config.region)

    run_id = config.run_id or new_run_id()
    archived_db = _archive_existing_dashboard_db(config.db_path, run_id=run_id)
    run_state = RunState.create(run_id=run_id, models=list(config.models))
    if archived_db is not None:
        run_state.extra["archived_db_dir"] = str(archived_db)
    run_state.instance_type = capacity.instance_type
    run_state.region = capacity.region
    run_state.status = "dry_run"
    run_state.save()
    print(f"Dry-run run state: {run_state.path()}")
    print(f"Dry-run selected {capacity.instance_type} in {capacity.region}")

    if config.dataset_path.exists():
        dataset = load_dataset(config.dataset_path)
        dataset_hash = dataset.content_hash
    else:
        dataset_hash = "dry-run"

    hourly = INSTANCE_HOURLY_USD.get(capacity.instance_type)

    store = BenchmarkStore(config.db_path)
    store.init_schema()
    store.insert_run(
        run_id=run_id,
        dataset_path=str(config.dataset_path),
        dataset_hash=dataset_hash,
        concurrency_levels=list(config.concurrency_levels),
        instance_type=capacity.instance_type,
        region=capacity.region,
        git_sha=git_sha(),
        archive_path=None,
        hourly_usd=hourly,
        node_count=len(config.models),
        status="dry_run",
    )
    store.finish_run(run_id, "dry_run")
    store.close()
    print(f"Dry-run DuckDB initialized: {config.db_path}")
    return 0


def wait_for_dashboard_control(run_id: str, db_path: Path) -> None:
    print("")
    print("Nodes are warm and ready. No measured dataset prompts have been sent.")
    print(f"Run ID: {run_id}")
    print(f"DuckDB: {db_path}")
    print(f"Dashboard: streamlit run {ROOT / 'scripts' / 'benchmark' / 'dashboard.py'}")
    print("Keep this process running while using the dashboard. Press Ctrl-C here to terminate nodes.")
    while True:
        time.sleep(3600)


def _record_failed_run(
    *,
    store: BenchmarkStore | None,
    db_path: Path,
    run_id: str,
    run_state: RunState,
    message: str,
) -> None:
    owns_store = store is None
    active_store = store or BenchmarkStore(db_path)
    try:
        active_store.insert_error(run_id=run_id, node_index=None, phase="orchestration", message=message)
        active_store.finish_run(run_id, "failed")
        run_state.status = "failed"
        run_state.save()
    finally:
        if owns_store:
            active_store.close()


def run_benchmark(config: BenchmarkConfig, *, wait_for_control_fn: WaitForControlFn | None = None) -> int:
    config.validate()
    validate_model_matrix(config.models)

    load_env_file(ENV_FILE)
    token = require("LAMBDA_CLOUD_API_KEY")
    api_key = require("MINISGL_API_KEY")
    hf_token = require("HF_TOKEN")
    ssh_key = Path(os.getenv("SSH_PRIVATE_KEY_PATH", str(Path.home() / ".ssh/id_ed25519")))

    run_id = config.run_id or new_run_id()
    archived_db = _archive_existing_dashboard_db(config.db_path, run_id=run_id)
    run_state = RunState.create(run_id=run_id, models=list(config.models))
    if archived_db is not None:
        run_state.extra["archived_db_dir"] = str(archived_db)
    store: BenchmarkStore | None = BenchmarkStore(config.db_path)
    wait_fn = wait_for_control_fn or wait_for_dashboard_control

    with BenchmarkCleanupManager(token=token, run_id=run_id, store=store, db_path=config.db_path) as cleanup:
        try:
            capacity = select_instance_type_with_capacity(token, preferred_region=config.region)
            print(f"Selected {capacity.instance_type} in {capacity.region}")

            ssh_key_name = _find_ssh_key_name(token, _read_public_key(ssh_key))
            launched = launch_benchmark_cluster(
                token,
                ssh_key_name,
                instance_type=capacity.instance_type,
                region=capacity.region,
                run_id=run_id,
            )
            instance_ids = [node.instance_id for node in launched]
            cleanup.register_instances(instance_ids)

            run_state.instance_type = capacity.instance_type
            run_state.region = capacity.region
            run_state.instance_ids = instance_ids
            for idx, node in enumerate(launched):
                run_state.nodes[idx].instance_id = node.instance_id
            run_state.status = "launched"
            run_state.save()

            ips = wait_for_instance_ips(token, instance_ids)
            archive = package_repo()
            sha = git_sha()
            run_state.git_sha = sha
            run_state.archive_path = str(archive)
            run_state.save()

            hourly = INSTANCE_HOURLY_USD.get(capacity.instance_type)

            from benchmark.dataset import load_dataset

            dataset = load_dataset(config.dataset_path)
            assert store is not None
            store.insert_run(
                run_id=run_id,
                dataset_path=str(config.dataset_path),
                dataset_hash=dataset.content_hash,
                concurrency_levels=list(config.concurrency_levels),
                instance_type=capacity.instance_type,
                region=capacity.region,
                git_sha=sha,
                archive_path=str(archive),
                hourly_usd=hourly,
                node_count=len(config.models),
                status="deploying",
            )

            deploy_results = _deploy_nodes_parallel(
                nodes=run_state.nodes,
                ips=ips,
                ssh_key=ssh_key,
                archive=archive,
                api_key=api_key,
                hf_token=hf_token,
                cuda_arch_list=CUDA_ARCH_BY_INSTANCE_TYPE.get(capacity.instance_type),
            )
            run_state.save()

            for node in run_state.nodes:
                if node.ip is None:
                    raise RuntimeError(f"Node {node.node_index} is missing an IP after deploy")
                instance_id = node.instance_id
                assert instance_id is not None
                ip = node.ip
                tunnel = start_ssh_tunnel(ip=ip, ssh_key=ssh_key, local_port=node.local_port)
                cleanup.register_tunnel_pid(tunnel.pid)
                node.tunnel_pid = tunnel.pid
                store.insert_node(
                    run_id=run_id,
                    node_index=node.node_index,
                    model_id=node.model_id,
                    instance_id=instance_id,
                    ip=ip,
                    local_port=node.local_port,
                    model_load_s=deploy_results[node.node_index],
                )

            run_state.status = "warming"
            run_state.save()
            store.update_run_status(run_id, "warming")
            asyncio.run(
                prepare_nodes_for_dashboard(
                    config=config,
                    run_state=run_state,
                    store=store,
                    api_key=api_key,
                    ssh_key=ssh_key,
                )
            )
            store.update_run_status(run_id, "ready")
            run_state.status = "ready"
            run_state.save()

            cleanup.store = None
            store.close()
            store = None
            wait_fn(run_id, config.db_path)
            return 0
        except Exception as exc:
            _record_failed_run(
                store=store,
                db_path=config.db_path,
                run_id=run_id,
                run_state=run_state,
                message=str(exc),
            )
            raise

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-model Lambda inference benchmark")
    parser.add_argument("--dataset", type=Path, default=BenchmarkConfig().dataset_path)
    parser.add_argument("--db", type=Path, default=BenchmarkConfig().db_path)
    parser.add_argument("--region", default=os.getenv("LAMBDA_REGION", "us-east-1"))
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=list(BenchmarkConfig().concurrency_levels),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        dataset_path=args.dataset,
        db_path=args.db,
        region=args.region,
        concurrency_levels=tuple(args.concurrency),
        dry_run=args.dry_run,
        run_id=args.run_id,
        models=MODEL_MATRIX,
    )

    if config.dry_run:
        return run_dry_run(config)
    return run_benchmark(config)


if __name__ == "__main__":
    raise SystemExit(main())

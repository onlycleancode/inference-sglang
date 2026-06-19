"""Lambda instance selection and multi-node launch."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from lambda_common import (  # noqa: E402
    api_request,
    instance_regions_with_capacity,
    pick_region,
    resolve_instance_types,
    terminate_instance,
)

from benchmark.config import BENCHMARK_INSTANCE_PREFIX, NODE_COUNT


@dataclass(frozen=True)
class SelectedCapacity:
    instance_type: str
    region: str


@dataclass(frozen=True)
class LaunchedNode:
    instance_id: str
    instance_type: str
    region: str
    name: str


def select_instance_type_with_capacity(
    token: str,
    *,
    preferred_region: str | None = None,
    instance_types: list[str] | None = None,
) -> SelectedCapacity:
    """Pick the first instance type with capacity, preferring H100 then A100."""
    ordered = instance_types or resolve_instance_types()
    last_error = ""
    for instance_type in ordered:
        regions = instance_regions_with_capacity(token, instance_type)
        if not regions:
            last_error = f"No capacity for {instance_type}"
            continue
        region = pick_region(token, instance_type, preferred_region)
        return SelectedCapacity(instance_type=instance_type, region=region)
    raise RuntimeError(f"No Lambda capacity for benchmark instance types. Last: {last_error}")


def launch_benchmark_cluster(
    token: str,
    ssh_key_name: str,
    *,
    instance_type: str,
    region: str,
    run_id: str,
    quantity: int = NODE_COUNT,
) -> list[LaunchedNode]:
    """Launch same-type instances in one region; caller must cleanup on failure.

    Lambda accounts can reject multi-instance launch requests with
    ``quantity > 1``. Launch nodes individually so the benchmark still gets a
    same-type, same-region cluster while keeping explicit partial-launch cleanup.
    """
    instance_ids: list[str] = []
    try:
        for index in range(quantity):
            payload = {
                "region_name": region,
                "instance_type_name": instance_type,
                "ssh_key_names": [ssh_key_name],
                "quantity": 1,
                "name": f"{BENCHMARK_INSTANCE_PREFIX}-{run_id}-{index}",
            }
            resp = api_request("POST", "/instance-operations/launch", token, payload)
            launched = resp.get("data", {}).get("instance_ids") or resp.get("instance_ids") or []
            if len(launched) != 1:
                for instance_id in launched:
                    terminate_instance(token, instance_id)
                raise RuntimeError(f"Expected 1 instance, got {len(launched)}: {resp}")
            instance_ids.append(launched[0])
    except BaseException:
        for instance_id in instance_ids:
            terminate_instance(token, instance_id)
        raise

    return [
        LaunchedNode(
            instance_id=instance_id,
            instance_type=instance_type,
            region=region,
            name=f"{BENCHMARK_INSTANCE_PREFIX}-{run_id}-{index}",
        )
        for index, instance_id in enumerate(instance_ids)
    ]


def wait_for_instance_ips(token: str, instance_ids: list[str], timeout_s: int = 900) -> dict[str, str]:
    """Poll Lambda until each instance has a public IP."""
    import time

    deadline = time.time() + timeout_s
    remaining = set(instance_ids)
    ips: dict[str, str] = {}
    while time.time() < deadline and remaining:
        instances = api_request("GET", "/instances", token).get("data", [])
        for item in instances:
            instance_id = item.get("id")
            if instance_id in remaining and item.get("ip"):
                ips[instance_id] = item["ip"]
                remaining.discard(instance_id)
        if remaining:
            time.sleep(10)
    if remaining:
        raise RuntimeError(f"Timed out waiting for IPs on instances: {sorted(remaining)}")
    return ips

"""Mandatory cleanup for benchmark Lambda instances and SSH tunnels."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from lambda_common import list_instances, terminate_instance  # noqa: E402

from benchmark.config import BENCHMARK_INSTANCE_PREFIX
from benchmark.duckdb_store import BenchmarkStore
from benchmark.run_state import RunState


class BenchmarkCleanupManager:
    """Track instance IDs and tunnel PIDs; always terminate on exit."""

    def __init__(
        self,
        *,
        token: str,
        run_id: str,
        store: BenchmarkStore | None = None,
        db_path: Path | None = None,
        terminate_fn: Callable[[str, str], None] | None = None,
        list_instances_fn: Callable[[str], list[dict]] | None = None,
    ) -> None:
        self.token = token
        self.run_id = run_id
        self.store = store
        self.db_path = db_path
        self._terminate = terminate_fn or terminate_instance
        self._list_instances = list_instances_fn or list_instances
        self.instance_ids: list[str] = []
        self.tunnel_pids: list[int] = []
        self._cleaned = False
        self._previous_handlers: dict[int, object] = {}

    def register_instances(self, instance_ids: list[str]) -> None:
        self.instance_ids = list(dict.fromkeys([*self.instance_ids, *instance_ids]))

    def register_tunnel_pid(self, pid: int) -> None:
        if pid not in self.tunnel_pids:
            self.tunnel_pids.append(pid)

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame) -> None:
        self.cleanup(reason=f"signal {signum}")
        raise SystemExit(128 + signum)

    def stop_tunnels(self) -> None:
        for pid in self.tunnel_pids:
            try:
                import os

                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.tunnel_pids.clear()

    def terminate_instances(self) -> None:
        for instance_id in self.instance_ids:
            try:
                self._terminate(self.token, instance_id)
            except SystemExit:
                # lambda_common raises SystemExit on API errors; keep terminating others.
                pass

    def verify_no_benchmark_instances(self) -> tuple[bool, list[str]]:
        """Poll Lambda and return remaining active/booting benchmark instances."""
        remaining: list[str] = []
        deadline = time.time() + 120
        while time.time() < deadline:
            remaining.clear()
            for item in self._list_instances(self.token):
                name = item.get("name") or ""
                status = item.get("status") or ""
                instance_id = item.get("id") or ""
                if (
                    instance_id in self.instance_ids
                    or name.startswith(BENCHMARK_INSTANCE_PREFIX)
                    or BENCHMARK_INSTANCE_PREFIX in name
                ) and status in {"active", "booting"}:
                    remaining.append(instance_id)
            if not remaining:
                return True, []
            time.sleep(10)
        return False, remaining

    def cleanup(self, *, reason: str = "finally") -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.stop_tunnels()
        self.terminate_instances()
        ok, remaining = self.verify_no_benchmark_instances()
        message = f"cleanup ({reason}): success={ok}, remaining={remaining}"
        try:
            if self.store is not None:
                self.store.insert_cleanup(
                    run_id=self.run_id,
                    instance_ids=self.instance_ids,
                    success=ok,
                    remaining_instances=remaining,
                    message=message,
                )
            elif self.db_path is not None:
                store = BenchmarkStore(self.db_path)
                try:
                    store.insert_cleanup(
                        run_id=self.run_id,
                        instance_ids=self.instance_ids,
                        success=ok,
                        remaining_instances=remaining,
                        message=message,
                    )
                finally:
                    store.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must prioritize termination.
            print(f"Warning: failed to record cleanup status: {exc}", file=sys.stderr)
        if not ok:
            raise RuntimeError(
                f"Benchmark cleanup failed: instances still active/booting: {remaining}"
            )

    def __enter__(self) -> BenchmarkCleanupManager:
        self.install_signal_handlers()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore_signal_handlers()
        self.cleanup(reason="finally" if exc is None else f"exception:{exc_type}")


def start_ssh_tunnel(
    *,
    ip: str,
    ssh_key: Path,
    local_port: int,
    remote_port: int = 1919,
) -> subprocess.Popen[bytes]:
    """Start a background SSH tunnel and return the subprocess."""
    cmd = [
        "ssh",
        "-N",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        f"ubuntu@{ip}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fetch_gpu_sample_via_ssh(ip: str, ssh_key: Path) -> tuple[float, float] | None:
    """Return (memory_used_mb, utilization_pct) from nvidia-smi over SSH."""
    cmd = [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=5",
        f"ubuntu@{ip}",
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])

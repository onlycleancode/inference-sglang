from __future__ import annotations

import pytest
import torch

from minisgl.server.args import ServerArgs, parse_args


def _parse(argv: list[str]) -> ServerArgs:
    server_args, _ = parse_args(argv)
    return server_args


def test_model_alias() -> None:
    args = _parse(["--model", "Qwen/Qwen3-8B", "--dtype", "bfloat16"])
    assert args.model_path == "Qwen/Qwen3-8B"


def test_model_path_long_form() -> None:
    args = _parse(["--model-path", "Qwen/Qwen3-8B", "--dtype", "bfloat16"])
    assert args.model_path == "Qwen/Qwen3-8B"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--tp", 4),
        ("--tp-size", 2),
        ("--tensor-parallel-size", 8),
    ],
)
def test_tensor_parallel_aliases(flag: str, value: int) -> None:
    args = _parse(["--model", "Qwen/Qwen3-8B", "--dtype", "bfloat16", flag, str(value)])
    assert args.tp_info.size == value


def test_host_and_port() -> None:
    args = _parse(
        [
            "--model",
            "Qwen/Qwen3-8B",
            "--dtype",
            "bfloat16",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
    )
    assert args.server_host == "0.0.0.0"
    assert args.server_port == 30000


def test_shell_mode_alias() -> None:
    args, run_shell = parse_args(["--model", "Qwen/Qwen3-0.6B", "--dtype", "bfloat16", "--shell"])
    assert run_shell is True
    assert args.cuda_graph_max_bs == 1
    assert args.max_running_req == 1
    assert args.silent_output is True


def test_shell_mode_long_form() -> None:
    args, run_shell = parse_args(
        ["--model", "Qwen/Qwen3-0.6B", "--dtype", "bfloat16", "--shell-mode"]
    )
    assert run_shell is True
    assert args.cuda_graph_max_bs == 1


def test_run_shell_parameter() -> None:
    args, run_shell = parse_args(["--model", "Qwen/Qwen3-0.6B", "--dtype", "bfloat16"], run_shell=True)
    assert run_shell is True
    assert args.max_running_req == 1


def test_dtype_mapping() -> None:
    args = _parse(["--model", "Qwen/Qwen3-8B", "--dtype", "float16"])
    assert args.dtype == torch.float16

#!/usr/bin/env python3
"""Call a remote Mini-SGLang OpenAI-compatible endpoint from your local machine."""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference against a remote Mini-SGLang server.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="User prompt text. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--base-url",
        default=_env("MINISGL_BASE_URL", "http://127.0.0.1:1919/v1"),
        help="OpenAI-compatible base URL (env: MINISGL_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=_env("MINISGL_API_KEY"),
        help="Bearer token / OpenAI api_key (env: MINISGL_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=_env("MINISGL_MODEL", "Qwen/Qwen3-8B"),
        help="Model id served by the remote server (env: MINISGL_MODEL).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(_env("MINISGL_MAX_TOKENS", "256") or "256"),
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=_truthy(_env("MINISGL_STREAM")),
        help="Stream tokens to stdout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(_env("MINISGL_TEMPERATURE", "0.7") or "0.7"),
    )
    return parser


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in ("1", "true", "yes")


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide a prompt argument or pipe text on stdin.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = read_prompt(args)

    if not args.api_key:
        print("MINISGL_API_KEY is required.", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.stream:
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
        print()
        return 0

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(response.choices[0].message.content or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

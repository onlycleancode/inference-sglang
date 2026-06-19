"""JSONL dataset parsing and stable hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetRow:
    row_index: int
    row_id: str
    prompt: str | None
    messages: list[dict[str, str]] | None
    max_tokens: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        if self.prompt is not None:
            return self.prompt
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in self.messages or []
        )

    def request_body(self, default_max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "max_tokens": self.max_tokens if self.max_tokens is not None else default_max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            "top_k": 1,
        }
        if self.messages is not None:
            body["messages"] = self.messages
        else:
            body["messages"] = [{"role": "user", "content": self.prompt or ""}]
        return body


@dataclass(frozen=True)
class Dataset:
    path: Path
    rows: tuple[DatasetRow, ...]
    content_hash: str

    @property
    def size(self) -> int:
        return len(self.rows)


def _stable_hash(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_row(raw: dict[str, Any], row_index: int) -> DatasetRow:
    prompt = raw.get("prompt")
    messages = raw.get("messages")
    if prompt is None and messages is None:
        raise ValueError(f"Line {row_index + 1}: require 'prompt' or 'messages'")
    if prompt is not None and messages is not None:
        raise ValueError(f"Line {row_index + 1}: provide only one of 'prompt' or 'messages'")
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Line {row_index + 1}: 'messages' must be a non-empty list")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Line {row_index + 1}: invalid message entry")

    row_id = str(raw.get("id", row_index))
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ValueError(f"Line {row_index + 1}: 'max_tokens' must be a positive integer")

    metadata = {
        key: value
        for key, value in raw.items()
        if key not in {"id", "prompt", "messages", "max_tokens"}
    }
    return DatasetRow(
        row_index=row_index,
        row_id=row_id,
        prompt=prompt,
        messages=messages,
        max_tokens=max_tokens,
        metadata=metadata,
    )


def load_dataset(path: Path) -> Dataset:
    """Load and validate a JSONL benchmark dataset."""
    if not path.exists():
        raise FileNotFoundError(path)

    raw_lines = path.read_text().splitlines()
    non_empty = [line for line in raw_lines if line.strip()]
    rows: list[DatasetRow] = []

    for index, line in enumerate(non_empty):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {index + 1}: malformed JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Line {index + 1}: each JSONL row must be an object")
        rows.append(_parse_row(payload, index))

    if not rows:
        raise ValueError("Dataset is empty")

    return Dataset(path=path, rows=tuple(rows), content_hash=_stable_hash(non_empty))


def warmup_request_body(prompt: str, *, max_tokens: int = 16) -> dict[str, Any]:
    """OpenAI chat payload for node warmup (top-level MiniSGLang sampling fields)."""
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "top_k": 1,
    }


def estimate_prompt_tokens(text: str) -> int:
    return max(1, len(text.split()))


def row_prompt_token_estimate(row: DatasetRow) -> int:
    hinted = row.metadata.get("prompt_token_estimate")
    if isinstance(hinted, int) and hinted > 0:
        return hinted
    return estimate_prompt_tokens(row.prompt_text)


def prompt_length_bucket(token_count: int) -> str:
    if token_count < 512:
        return "<512"
    if token_count < 4096:
        return "512-4k"
    if token_count < 8192:
        return "4k-8k"
    if token_count < 16384:
        return "8k-16k"
    return "16k+"


def output_length_bucket(token_count: int) -> str:
    if token_count < 32:
        return "<32"
    if token_count < 128:
        return "32-128"
    if token_count < 512:
        return "128-512"
    return "512+"

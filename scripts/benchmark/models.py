"""Model matrix helpers."""

from __future__ import annotations

from benchmark.config import MODEL_MATRIX, NODE_COUNT


def validate_model_matrix(models: tuple[tuple[str, int], ...] | None = None) -> None:
    """Ensure the benchmark uses exactly three distinct MiniSGLang models."""
    matrix = models or MODEL_MATRIX
    if len(matrix) != NODE_COUNT:
        raise ValueError(f"Model matrix must contain {NODE_COUNT} entries")
    model_ids = [model_id for model_id, _ in matrix]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Model matrix entries must be unique")
    ports = [port for _, port in matrix]
    if len(set(ports)) != len(ports):
        raise ValueError("Tunnel ports must be unique")

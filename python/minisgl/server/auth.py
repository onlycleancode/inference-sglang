from __future__ import annotations

import os
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from minisgl.utils import init_logger

logger = init_logger(__name__)

# Routes that require a valid bearer token when auth is enabled.
PROTECTED_PATHS = frozenset(
    {
        "/generate",
        "/v1/chat/completions",
        "/v1/models",
        "/admin/shutdown",
    }
)

# Unauthenticated health checks for load balancers and smoke scripts.
PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/status",
        "/v1",
    }
)


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in ("1", "true", "yes")


def load_auth_settings() -> tuple[str | None, bool, bool]:
    """Return (api_key, auth_disabled, require_api_key)."""
    api_key = os.getenv("MINISGL_API_KEY") or None
    auth_disabled = _truthy(os.getenv("MINISGL_AUTH_DISABLED"))
    require_api_key = _truthy(os.getenv("MINISGL_REQUIRE_API_KEY"))
    return api_key, auth_disabled, require_api_key


def validate_auth_at_startup(*, run_shell: bool) -> None:
    api_key, auth_disabled, require_api_key = load_auth_settings()

    if run_shell:
        return

    if require_api_key and not api_key:
        raise SystemExit(
            "MINISGL_REQUIRE_API_KEY is set but MINISGL_API_KEY is missing. "
            "Set MINISGL_API_KEY in the environment before starting the server."
        )

    if not api_key and not auth_disabled:
        logger.warning(
            "MINISGL_API_KEY is unset; API auth is disabled for local development. "
            "Set MINISGL_AUTH_DISABLED=1 to silence this warning, or set MINISGL_API_KEY "
            "for deployed endpoints."
        )


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None

    # OpenAI Python client also supports api_key via x-api-key on some providers.
    api_key_header = request.headers.get("x-api-key")
    if api_key_header:
        return api_key_header.strip() or None

    return None


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        api_key: str | None,
        auth_disabled: bool,
    ) -> None:
        super().__init__(app)
        self._api_key = api_key
        self._auth_disabled = auth_disabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        if request.method == "OPTIONS" or path in PUBLIC_PATHS:
            return await call_next(request)

        if path not in PROTECTED_PATHS:
            return await call_next(request)

        if self._auth_disabled or not self._api_key:
            return await call_next(request)

        token = _extract_bearer_token(request)
        if token != self._api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)

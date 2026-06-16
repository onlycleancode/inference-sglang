from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from minisgl.server.auth import (
    BearerAuthMiddleware,
    load_auth_settings,
    validate_auth_at_startup,
)


@pytest.fixture
def auth_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        BearerAuthMiddleware,
        api_key="test-secret-key",
        auth_disabled=False,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1")
    def v1_root():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"data": []}

    @app.post("/v1/chat/completions")
    def chat():
        return {"choices": []}

    @app.post("/generate")
    def generate():
        return {"ok": True}

    return app


def test_health_is_public(auth_app: FastAPI) -> None:
    client = TestClient(auth_app)
    response = client.get("/health")
    assert response.status_code == 200


def test_v1_root_is_public(auth_app: FastAPI) -> None:
    client = TestClient(auth_app)
    response = client.get("/v1")
    assert response.status_code == 200


def test_missing_token_returns_401(auth_app: FastAPI) -> None:
    client = TestClient(auth_app)
    response = client.get("/v1/models")
    assert response.status_code == 401


def test_wrong_token_returns_401(auth_app: FastAPI) -> None:
    client = TestClient(auth_app)
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401


def test_valid_bearer_token(auth_app: FastAPI) -> None:
    client = TestClient(auth_app)
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code == 200


def test_valid_x_api_key_header(auth_app: FastAPI) -> None:
    client = TestClient(auth_app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-api-key": "test-secret-key"},
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200


def test_auth_disabled_allows_open_access() -> None:
    app = FastAPI()
    app.add_middleware(
        BearerAuthMiddleware,
        api_key="test-secret-key",
        auth_disabled=True,
    )

    @app.get("/v1/models")
    def models():
        return {"data": []}

    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200


def test_no_api_key_disables_enforcement() -> None:
    app = FastAPI()
    app.add_middleware(
        BearerAuthMiddleware,
        api_key=None,
        auth_disabled=False,
    )

    @app.get("/v1/models")
    def models():
        return {"data": []}

    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200


def test_load_auth_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINISGL_API_KEY", "from-env")
    monkeypatch.setenv("MINISGL_AUTH_DISABLED", "1")
    monkeypatch.setenv("MINISGL_REQUIRE_API_KEY", "true")

    api_key, auth_disabled, require_api_key = load_auth_settings()
    assert api_key == "from-env"
    assert auth_disabled is True
    assert require_api_key is True


def test_validate_auth_exits_when_required_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINISGL_API_KEY", raising=False)
    monkeypatch.setenv("MINISGL_REQUIRE_API_KEY", "1")

    with pytest.raises(SystemExit):
        validate_auth_at_startup(run_shell=False)


def test_validate_auth_skips_in_shell_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINISGL_API_KEY", raising=False)
    monkeypatch.setenv("MINISGL_REQUIRE_API_KEY", "1")
    validate_auth_at_startup(run_shell=True)

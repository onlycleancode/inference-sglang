from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from minisgl.server.auth import BearerAuthMiddleware


@pytest.fixture
def admin_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        BearerAuthMiddleware,
        api_key="test-secret-key",
        auth_disabled=False,
    )

    @app.get("/status")
    def status():
        return {"status": "ok"}

    @app.post("/admin/shutdown")
    def shutdown():
        return {"status": "shutting_down"}

    return app


def test_status_is_public(admin_app: FastAPI) -> None:
    client = TestClient(admin_app)
    response = client.get("/status")
    assert response.status_code == 200


def test_admin_shutdown_requires_token(admin_app: FastAPI) -> None:
    client = TestClient(admin_app)
    response = client.post("/admin/shutdown", json={})
    assert response.status_code == 401


def test_admin_shutdown_accepts_token(admin_app: FastAPI) -> None:
    client = TestClient(admin_app)
    response = client.post(
        "/admin/shutdown",
        json={},
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code == 200

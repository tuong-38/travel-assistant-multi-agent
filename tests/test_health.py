from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import health
from app.main import create_app


class FakeConnection:
    async def execute(self, query: str) -> None:
        assert query == "SELECT 1"


class FakeConnectionContext:
    async def __aenter__(self) -> FakeConnection:
        return FakeConnection()

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePool:
    def connection(self) -> FakeConnectionContext:
        return FakeConnectionContext()


def test_live_health() -> None:
    app = create_app()
    response = TestClient(app).get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_health_with_database(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            assert url == "http://litellm:4000/health/readiness"
            return FakeResponse()

    app = create_app()
    app.state.db_pool = FakePool()
    monkeypatch.setattr(
        health, "get_settings", lambda: SimpleNamespace(litellm_base_url="http://litellm:4000/v1")
    )
    monkeypatch.setattr(health.httpx, "AsyncClient", lambda timeout: FakeClient())
    response = TestClient(app).get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_health_when_litellm_is_unavailable(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            request = httpx.Request("GET", "http://litellm:4000/health/readiness")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("unavailable", request=request, response=response)

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        health, "get_settings", lambda: SimpleNamespace(litellm_base_url="http://litellm:4000/v1")
    )
    monkeypatch.setattr(health.httpx, "AsyncClient", lambda timeout: FakeClient())
    app = create_app()
    app.state.db_pool = FakePool()
    response = TestClient(app).get("/api/v1/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}

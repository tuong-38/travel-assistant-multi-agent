from fastapi.testclient import TestClient

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


def test_ready_health_with_database() -> None:
    app = create_app()
    app.state.db_pool = FakePool()
    response = TestClient(app).get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

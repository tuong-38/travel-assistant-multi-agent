from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import create_app


class FakeGraph:
    def __init__(self) -> None:
        self.thread_ids: list[str] = []
        self.model_aliases: list[str] = []

    async def ainvoke(self, state: dict, config: dict) -> dict:
        thread_id = config["configurable"]["thread_id"]
        self.thread_ids.append(thread_id)
        self.model_aliases.append(config["configurable"]["model_alias"])
        message = state["messages"][-1]["content"]
        return {"messages": [{"role": "assistant", "content": f"You said: {message}"}]}


def test_chat_generates_thread_id(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_api, "get_settings", lambda: SimpleNamespace(default_model="travel_general")
    )
    app = create_app()
    graph = FakeGraph()
    app.state.graph = graph
    response = TestClient(app).post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 200
    result = response.json()
    assert result["reply"] == "You said: Hello"
    assert UUID(result["thread_id"]).__str__() == graph.thread_ids[0]
    assert graph.model_aliases == ["travel_general"]


def test_chat_uses_supplied_thread_id() -> None:
    app = create_app()
    graph = FakeGraph()
    app.state.graph = graph
    thread_id = "9de6a955-31ee-4a0c-b877-34123cd130d5"
    response = TestClient(app).post(
        "/api/v1/chat", json={"message": "Plan a trip", "thread_id": thread_id}
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id
    assert graph.thread_ids == [thread_id]


def test_chat_uses_requested_model() -> None:
    app = create_app()
    graph = FakeGraph()
    app.state.graph = graph
    response = TestClient(app).post(
        "/api/v1/chat", json={"message": "Xin chào", "model": "travel_local"}
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "You said: Xin chào"
    assert graph.model_aliases == ["travel_local"]

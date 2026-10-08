import asyncio
from json import dumps, loads
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.workflow as workflow
from app.api import chat as chat_api
from app.api.chat import _thread_resume_lock
from app.graph.workflow import build_graph
from app.main import create_app


class HitlFakeModel:
    def __init__(self) -> None:
        self.supervisor_calls = 0
        self.planner_calls = 0
        self.itinerary_calls = 0

    async def ainvoke(self, messages):
        prompt = messages[0].content
        if "Choose exactly one next step" in prompt:
            route = ["destination", "planner", "FINISH"][self.supervisor_calls]
            self.supervisor_calls += 1
            if route == "FINISH":
                return AIMessage(content='{"next":"FINISH","response":"Trip complete"}')
            return AIMessage(content=f'{{"next":"{route}"}}')
        if "Extract only the destination" in prompt:
            return AIMessage(content='{"destination":"Hanoi"}')
        if "using only facts in the structured lookup result" in prompt:
            return AIMessage(content="Hanoi information.")
        if "You are the travel planner" in prompt:
            self.planner_calls += 1
            request = loads(messages[1].content)
            plan = request["travel_plan"] or {
                "destination": "Hanoi",
                "dates": "2026-11-01 to 2026-11-03",
                "duration": "3 days",
                "budget": "moderate",
                "preferences": ["food"],
                "constraints": [],
            }
            plan.update(request["plan_revision_feedback"] or {})
            return AIMessage(content=dumps(plan))
        if "You are the itinerary specialist" in prompt:
            self.itinerary_calls += 1
            return AIMessage(content="Day 1: Hanoi")
        raise AssertionError(f"Unexpected prompt: {prompt}")


class CountingMCP:
    def __init__(self) -> None:
        self.calls = 0

    async def search_destination(self, destination: str):
        assert destination == "Hanoi"
        self.calls += 1
        return {
            "destination": "Hanoi",
            "found": True,
            "summary": "Historic city.",
            "activities": ["Old Quarter"],
        }


def _hitl_client(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: SimpleNamespace(
            default_model="travel_general",
            travel_tools_mcp_url="http://unused",
            mcp_timeout_seconds=5.0,
        ),
    )
    monkeypatch.setattr(
        chat_api, "get_settings", lambda: SimpleNamespace(default_model="travel_general")
    )
    model = HitlFakeModel()
    mcp_client = CountingMCP()
    graph = build_graph(InMemorySaver(), chat_model=model, mcp_client=mcp_client)
    app = create_app()
    app.state.graph = graph
    app.state.testing = True
    return TestClient(app), model, mcp_client, graph


class FakeConnection:
    def __init__(self, *, fail_unlock: bool = False) -> None:
        self.fail_unlock = fail_unlock
        self.queries: list[str] = []
        self.closed = False

    async def execute(self, query: str, params: tuple[str, ...]) -> None:
        self.queries.append(query)
        assert params == ("lock-test-thread",)
        if "pg_advisory_unlock" in query and self.fail_unlock:
            raise RuntimeError("unlock failed")

    async def close(self) -> None:
        self.closed = True


class FakePoolConnection:
    def __init__(self, pool: "FakePool") -> None:
        self.pool = pool

    async def __aenter__(self) -> FakeConnection:
        self.pool.acquired += 1
        return self.pool.connection_object

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        if self.pool.connection_object.closed:
            self.pool.discarded += 1
        else:
            self.pool.returned += 1
        return False


class FakePool:
    def __init__(self, *, fail_unlock: bool = False) -> None:
        self.connection_object = FakeConnection(fail_unlock=fail_unlock)
        self.acquired = 0
        self.returned = 0
        self.discarded = 0

    def connection(self) -> FakePoolConnection:
        return FakePoolConnection(self)


def _lock_test_request(*, pool: FakePool | None = None, testing: bool = False):
    app = create_app()
    app.state.db_pool = pool
    app.state.testing = testing
    return SimpleNamespace(app=app)


async def test_advisory_lock_success_returns_connection_to_pool() -> None:
    pool = FakePool()
    async with _thread_resume_lock(_lock_test_request(pool=pool), "lock-test-thread"):
        pass

    assert pool.acquired == 1
    assert pool.returned == 1
    assert pool.discarded == 0
    assert len(pool.connection_object.queries) == 2
    assert "pg_advisory_lock" in pool.connection_object.queries[0]
    assert "pg_advisory_unlock" in pool.connection_object.queries[1]


async def test_advisory_lock_preserves_workflow_exception_and_unlocks() -> None:
    pool = FakePool()
    with pytest.raises(ValueError, match="workflow failed"):
        async with _thread_resume_lock(_lock_test_request(pool=pool), "lock-test-thread"):
            raise ValueError("workflow failed")

    assert pool.connection_object.queries[-1].find("pg_advisory_unlock") >= 0
    assert pool.returned == 1
    assert pool.discarded == 0


@pytest.mark.parametrize("workflow_fails", [False, True])
async def test_advisory_unlock_failure_discards_connection_and_preserves_work_error(
    workflow_fails: bool,
) -> None:
    pool = FakePool(fail_unlock=True)
    expected_error = ValueError("workflow failed")
    if workflow_fails:
        with pytest.raises(ValueError, match="workflow failed"):
            async with _thread_resume_lock(_lock_test_request(pool=pool), "lock-test-thread"):
                raise expected_error
    else:
        with pytest.raises(HTTPException) as error:
            async with _thread_resume_lock(_lock_test_request(pool=pool), "lock-test-thread"):
                pass
        assert error.value.status_code == 503

    assert pool.connection_object.closed is True
    assert pool.returned == 0
    assert pool.discarded == 1


def test_resume_without_production_pool_fails_closed(monkeypatch) -> None:
    class ForbiddenThreadLocks:
        def setdefault(self, key, default):
            raise AssertionError("production must not use process-local locking")

    monkeypatch.setattr(chat_api, "_thread_locks", ForbiddenThreadLocks())
    app = create_app()
    app.state.graph = object()

    response = TestClient(app).post(
        "/api/v1/chat/9de6a955-31ee-4a0c-b877-34123cd130d5/resume",
        json={"interrupt_id": "unused", "decision": "approve"},
    )

    assert response.status_code == 503


def test_modify_limit_and_approve_preserve_thread(monkeypatch) -> None:
    client, model, mcp_client, graph = _hitl_client(monkeypatch)
    thread_id = "9de6a955-31ee-4a0c-b877-34123cd130d5"
    initial = client.post(
        "/api/v1/chat",
        json={"message": "Plan a trip to Hanoi", "model": "travel_local", "thread_id": thread_id},
    )
    assert initial.status_code == 200
    pending = initial.json()["pending_approval"]
    assert initial.json()["status"] == "interrupted"
    assert pending["travel_plan"]["destination"] == "Hanoi"
    assert mcp_client.calls == 1

    pending_chat = client.post(
        "/api/v1/chat", json={"message": "Another request", "thread_id": thread_id}
    )
    assert pending_chat.status_code == 409

    interrupt_id = pending["interrupt_id"]
    original_interrupt_id = interrupt_id
    for revision, budget in enumerate(("moderate", "premium", "luxury"), start=1):
        modified = client.post(
            f"/api/v1/chat/{thread_id}/resume",
            json={
                "interrupt_id": interrupt_id,
                "decision": "modify",
                "changes": {"budget": budget},
            },
        )
        assert modified.status_code == 200, modified.text
        body = modified.json()
        assert body["pending_approval"]["plan_revision_count"] == revision
        assert body["pending_approval"]["travel_plan"]["budget"] == budget
        if revision == 1:
            stale = client.post(
                f"/api/v1/chat/{thread_id}/resume",
                json={"interrupt_id": original_interrupt_id, "decision": "approve"},
            )
            assert stale.status_code == 409
        interrupt_id = body["pending_approval"]["interrupt_id"]

    capped = client.post(
        f"/api/v1/chat/{thread_id}/resume",
        json={"interrupt_id": interrupt_id, "decision": "modify", "changes": {"budget": "low"}},
    )
    assert capped.status_code == 409

    approved = client.post(
        f"/api/v1/chat/{thread_id}/resume",
        json={"interrupt_id": interrupt_id, "decision": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json() == {"thread_id": thread_id, "reply": "Day 1: Hanoi"}
    assert model.planner_calls == 4
    assert model.itinerary_calls == 1
    assert model.supervisor_calls == 2
    assert mcp_client.calls == 1

    replay = client.post(
        f"/api/v1/chat/{thread_id}/resume",
        json={"interrupt_id": interrupt_id, "decision": "approve"},
    )
    assert replay.status_code == 409
    assert model.itinerary_calls == 1
    snapshot = asyncio.run(graph.aget_state({"configurable": {"thread_id": thread_id}}))
    assert snapshot.values["model_alias"] == "travel_local"
    assert snapshot.values["user_request"] == "Plan a trip to Hanoi"
    assert [item["content"] for item in snapshot.values["messages"] if item["role"] == "user"] == [
        "Plan a trip to Hanoi"
    ]


def test_reject_ends_without_replanning_or_itinerary(monkeypatch) -> None:
    client, model, mcp_client, _ = _hitl_client(monkeypatch)
    initial = client.post("/api/v1/chat", json={"message": "Plan a trip to Hanoi"})
    thread_id = initial.json()["thread_id"]
    interrupt_id = initial.json()["pending_approval"]["interrupt_id"]
    rejected = client.post(
        f"/api/v1/chat/{thread_id}/resume",
        json={"interrupt_id": interrupt_id, "decision": "reject"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert model.planner_calls == 1
    assert model.itinerary_calls == 0
    assert model.supervisor_calls == 2
    assert mcp_client.calls == 1

    again = client.post(
        f"/api/v1/chat/{thread_id}/resume",
        json={"interrupt_id": interrupt_id, "decision": "approve"},
    )
    assert again.status_code == 409


def test_resume_validation_and_unknown_thread(monkeypatch) -> None:
    client, _, _, _ = _hitl_client(monkeypatch)
    unknown = client.post(
        "/api/v1/chat/9de6a955-31ee-4a0c-b877-34123cd130d5/resume",
        json={"interrupt_id": "missing", "decision": "unknown"},
    )
    assert unknown.status_code == 404

    initial = client.post("/api/v1/chat", json={"message": "Plan a trip to Hanoi"})
    thread_id = initial.json()["thread_id"]
    interrupt_id = initial.json()["pending_approval"]["interrupt_id"]
    invalid_payloads = [
        ({"interrupt_id": interrupt_id, "decision": "unknown"}, 422),
        ({"interrupt_id": interrupt_id, "decision": "modify"}, 422),
        ({"interrupt_id": interrupt_id, "decision": "modify", "changes": {}}, 422),
        ({"interrupt_id": interrupt_id, "decision": "modify", "changes": {"unknown": "x"}}, 422),
        ({"interrupt_id": interrupt_id, "decision": "approve", "changes": {"budget": "low"}}, 422),
        ({"interrupt_id": interrupt_id, "decision": "reject", "changes": {"budget": "low"}}, 422),
        (
            {
                "interrupt_id": interrupt_id,
                "decision": "modify",
                "changes": {"preferences": "food"},
            },
            422,
        ),
        ({"interrupt_id": interrupt_id, "decision": "approve", "extra": True}, 422),
        ({"interrupt_id": "stale", "decision": "approve"}, 409),
    ]
    for body, expected_status in invalid_payloads:
        response = client.post(f"/api/v1/chat/{thread_id}/resume", json=body)
        assert response.status_code == expected_status

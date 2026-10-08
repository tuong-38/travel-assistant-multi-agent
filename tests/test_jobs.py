import asyncio
import json
import operator
from types import SimpleNamespace
from typing import Annotated, TypedDict
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from structlog.testing import capture_logs

import app.api.jobs as jobs_api
from app.api.jobs import router as jobs_router
from app.graph.workflow import WorkflowRoutingError
from app.jobs.queue import JobQueue
from app.worker import StreamWorker


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def xadd(self, stream, fields):
        self.commands.append(("xadd", stream, fields))
        return self

    def hset(self, key, mapping):
        self.commands.append(("hset", key, mapping))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def xack(self, stream, group, stream_id):
        self.commands.append(("xack", stream, group, stream_id))
        return self

    def xdel(self, stream, stream_id):
        self.commands.append(("xdel", stream, stream_id))
        return self

    async def execute(self):
        result = []
        for command in self.commands:
            if command[0] == "xadd":
                _, stream, fields = command
                self.redis.stream_entries.append(("1-0", fields))
                result.append("1-0")
            elif command[0] == "hset":
                _, key, mapping = command
                self.redis.hashes.setdefault(key, {}).update(mapping)
                result.append(1)
            elif command[0] in {"xack", "xdel"}:
                self.redis.calls.append((command[0], command[1:], {}))
                result.append(1)
            else:
                result.append(True)
        return result


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.stream_entries = []
        self.calls = []

    async def xgroup_create(self, *args, **kwargs):
        self.calls.append(("xgroup_create", args, kwargs))

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    async def hgetall(self, key):
        return self.hashes.get(key, {})

    async def expire(self, *args):
        return True

    async def xreadgroup(self, *args, **kwargs):
        return [("travel:jobs", self.stream_entries)] if self.stream_entries else []

    async def xautoclaim(self, *args, **kwargs):
        self.calls.append(("xautoclaim", args, kwargs))
        return ["0-0", self.stream_entries, []]

    async def xpending_range(self, *args, **kwargs):
        return [{"times_delivered": 2}]

    async def xclaim(self, *args, **kwargs):
        self.calls.append(("xclaim", args, kwargs))

    async def aclose(self):
        return None


def test_job_queue_stream_enqueue_recovery_and_ack() -> None:
    redis = FakeRedis()
    queue = JobQueue(redis, stream="travel:jobs", group="workers", status_ttl_seconds=60)
    job_id = UUID("9de6a955-31ee-4a0c-b877-34123cd130d5")
    thread_id = UUID("7a4d1994-9521-4b7d-9095-c1a6a443da05")

    async def exercise():
        await queue.initialize()
        await queue.enqueue(
            job_id=job_id,
            thread_id=thread_id,
            operation="chat",
            payload={"message": "hello", "model_alias": "travel_general"},
        )
        job = await queue.get(str(job_id))
        claimed = await queue.claim_one("worker-1", min_idle_ms=1000)
        deliveries = await queue.deliveries("1-0")
        await queue.heartbeat("worker-1", "1-0")
        await queue.acknowledge("1-0")
        return job, claimed, deliveries

    job, claimed, deliveries = asyncio.run(exercise())
    assert job is not None and job["status"] == "queued"
    assert job["operation"] == "chat"
    assert json.loads(job["payload"])["message"] == "hello"
    assert claimed[0][0] == "1-0"
    assert deliveries == 2
    assert any(call[0] == "xautoclaim" for call in redis.calls)
    assert any(call[0] == "xclaim" for call in redis.calls)
    assert any(call[0] == "xack" for call in redis.calls)


class FakePoolConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, *_):
        return None

    async def close(self):
        return None


class FakePool:
    def connection(self):
        return FakePoolConnection()


class FakeQueue:
    def __init__(self):
        self.jobs = {}
        self.entries = []
        self.acks = []
        self.delivery_count = 1

    async def enqueue(self, *, job_id, thread_id, operation, payload):
        job = {
            "job_id": str(job_id),
            "thread_id": str(thread_id),
            "operation": operation,
            "payload": json.dumps(payload),
            "status": "queued",
        }
        self.jobs[str(job_id)] = job
        fields = {**job, "payload": job["payload"]}
        self.entries.append((str(len(self.entries) + 1) + "-0", fields))

    async def get(self, job_id):
        return self.jobs.get(job_id)

    async def restore_status(self, fields):
        self.jobs[fields["job_id"]] = {**fields, "status": "queued"}

    async def set_status(self, job_id, status, *, error=""):
        self.jobs[job_id]["status"] = status
        self.jobs[job_id]["error"] = error

    async def deliveries(self, _):
        return self.delivery_count

    async def acknowledge(self, stream_id):
        self.acks.append(stream_id)

    async def heartbeat(self, *_):
        return None


class InterruptState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], operator.add]
    user_request: str
    model_alias: str
    execution_job_id: str | None
    decision: str
    completed_agents: list[str]
    destination_info: dict | None
    travel_plan: dict | None
    itinerary: str | None
    current_agent: str | None
    final_response: str | None
    hitl_decision: str | None
    plan_revision_count: int
    plan_revision_feedback: dict | None


def _approval_graph():
    async def approval(_state):
        value = interrupt({"type": "approval"})
        return {"decision": value["decision"]}

    builder = StateGraph(InterruptState)
    builder.add_node("approval", approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=InMemorySaver())


def _jobs_test_app(graph, queue):
    app = FastAPI()
    app.include_router(jobs_router, prefix="/api/v1")
    app.state.graph = graph
    app.state.jobs = queue
    app.state.db_pool = FakePool()
    app.state.testing = True
    return app


def test_async_chat_enqueues_without_running_graph(monkeypatch) -> None:
    class UntouchedGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={}, next=(), interrupts=())

        async def ainvoke(self, *_args, **_kwargs):
            raise AssertionError("API enqueue must not execute the graph")

    monkeypatch.setattr(
        jobs_api, "get_settings", lambda: SimpleNamespace(default_model="travel_general")
    )
    queue = FakeQueue()
    app = _jobs_test_app(UntouchedGraph(), queue)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/jobs",
            json={"message": "Tell me about Hanoi", "model": "travel_local"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert UUID(body["job_id"])
        assert UUID(body["thread_id"])
        job = queue.jobs[body["job_id"]]
        assert job["operation"] == "chat"
        payload = json.loads(job["payload"])
        assert payload == {"message": "Tell me about Hanoi", "model_alias": "travel_local"}
        status_response = client.get(f"/api/v1/jobs/{body['job_id']}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "queued"


def test_async_resume_duplicate_is_revalidated_against_checkpoint() -> None:
    graph = _approval_graph()
    thread_id = "95a71310-396c-4c09-a178-7c95cc042652"
    config = {"configurable": {"thread_id": thread_id}}
    asyncio.run(graph.ainvoke({}, config=config))
    queue = FakeQueue()
    client = TestClient(_jobs_test_app(graph, queue))
    payload = {"interrupt_id": "1", "decision": "approve"}

    # Obtain the real interrupt id from the checkpoint rather than assuming its value.
    snapshot = asyncio.run(graph.aget_state(config))
    payload["interrupt_id"] = snapshot.interrupts[0].id
    first = client.post(f"/api/v1/chat/{thread_id}/resume/jobs", json=payload)
    second = client.post(f"/api/v1/chat/{thread_id}/resume/jobs", json=payload)
    assert first.status_code == second.status_code == 202
    assert len(queue.entries) == 2  # No interrupt reservation key is used.

    settings = SimpleNamespace(
        redis_claim_idle_ms=300_000,
        redis_heartbeat_seconds=30.0,
        job_max_delivery_attempts=5,
    )
    worker = StreamWorker(queue, graph, FakePool(), settings)
    first_id, first_fields = queue.entries[0]
    asyncio.run(worker._handle("worker-1", first_id, first_fields))
    second_id, second_fields = queue.entries[1]
    asyncio.run(worker._handle("worker-1", second_id, second_fields))

    assert queue.jobs[first_fields["job_id"]]["status"] == "succeeded"
    assert queue.jobs[second_fields["job_id"]]["status"] == "failed"
    assert "stale" in queue.jobs[second_fields["job_id"]]["error"].lower()
    final_snapshot = asyncio.run(graph.aget_state(config))
    assert final_snapshot.values["decision"] == "approve"
    assert len(queue.acks) == 2


def test_worker_redelivery_continues_after_checkpoint_without_replaying_node() -> None:
    calls = {"first": 0, "second": 0}
    fail_once = {"value": True}
    partial_states = []

    async def first_node(_state):
        calls["first"] += 1
        return {
            "decision": "ready",
            "completed_agents": ["destination"],
            "destination_info": {
                "destination": "Hanoi",
                "found": True,
                "summary": "Historic city.",
                "activities": [],
            },
            "travel_plan": {"destination": "Hanoi"},
        }

    async def second_node(state):
        calls["second"] += 1
        partial_states.append(
            (state["completed_agents"], state["destination_info"], state["travel_plan"])
        )
        if fail_once["value"]:
            fail_once["value"] = False
            raise RuntimeError("simulated failure during node execution")
        return {"decision": state["decision"] + ":done"}

    builder = StateGraph(InterruptState)
    builder.add_node("first", first_node)
    builder.add_node("second", second_node)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    queue = FakeQueue()
    thread_id = "27ef60f0-f938-4fc2-8496-fba3e55c8a5c"
    job_id = "2b1b890e-6624-43ee-b0bd-fc2211cc552f"
    job = {
        "job_id": job_id,
        "thread_id": thread_id,
        "operation": "chat",
        "payload": json.dumps({"message": "hello", "model_alias": "travel_general"}),
        "status": "running",
    }
    queue.jobs[job_id] = job
    worker = StreamWorker(
        queue,
        graph,
        FakePool(),
        SimpleNamespace(
            redis_claim_idle_ms=300_000,
            redis_heartbeat_seconds=30.0,
            job_max_delivery_attempts=5,
        ),
    )

    async def recover():
        with pytest.raises(RuntimeError, match="simulated failure"):
            await worker._execute(job)
        await worker._execute(job)
        config = {"configurable": {"thread_id": thread_id}}
        return await graph.aget_state(config)

    snapshot = asyncio.run(recover())
    assert calls == {"first": 1, "second": 2}
    assert (
        partial_states
        == [
            (
                ["destination"],
                {
                    "destination": "Hanoi",
                    "found": True,
                    "summary": "Historic city.",
                    "activities": [],
                },
                {"destination": "Hanoi"},
            )
        ]
        * 2
    )
    assert snapshot.values["execution_job_id"] == job_id
    assert snapshot.values["decision"] == "ready:done"
    assert [message["content"] for message in snapshot.values["messages"]] == ["hello"]


def test_worker_new_chat_job_starts_fresh_on_same_thread() -> None:
    observed_states = []

    async def execute(state):
        observed_states.append(state.copy())
        if state["user_request"] == "Request A":
            return {
                "completed_agents": ["destination", "planner", "itinerary"],
                "destination_info": {"destination": "Hanoi", "found": True},
                "travel_plan": {"destination": "Hanoi"},
                "itinerary": "Old itinerary",
                "current_agent": "FINISH",
                "final_response": "Old itinerary",
                "hitl_decision": "approve",
                "plan_revision_count": 2,
                "plan_revision_feedback": {"budget": "premium"},
                "messages": [{"role": "assistant", "content": "Old itinerary"}],
            }
        return {
            "decision": "Request B executed",
            "messages": [{"role": "assistant", "content": "Follow-up received."}],
        }

    builder = StateGraph(InterruptState)
    builder.add_node("execute", execute)
    builder.add_edge(START, "execute")
    builder.add_edge("execute", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    queue = FakeQueue()
    worker = StreamWorker(
        queue,
        graph,
        FakePool(),
        SimpleNamespace(
            redis_claim_idle_ms=300_000,
            redis_heartbeat_seconds=30.0,
            job_max_delivery_attempts=5,
        ),
    )
    thread_id = "continued-worker-thread"

    async def run_job(job_id, message):
        job = {
            "job_id": job_id,
            "thread_id": thread_id,
            "operation": "chat",
            "payload": json.dumps({"message": message, "model_alias": "travel_local"}),
            "status": "queued",
            "error": "",
        }
        queue.jobs[job_id] = job
        await worker._handle("worker-1", f"{job_id}-0", job)
        return job

    async def exercise():
        first = await run_job("job-A", "Request A")
        second = await run_job("job-B", "Request B")
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        return first, second, snapshot

    first, second, snapshot = asyncio.run(exercise())
    assert first["status"] == second["status"] == "succeeded"
    assert observed_states[0]["completed_agents"] == []
    assert observed_states[1]["completed_agents"] == []
    assert observed_states[1]["destination_info"] is None
    assert observed_states[1]["travel_plan"] is None
    assert observed_states[1]["itinerary"] is None
    assert observed_states[1]["final_response"] is None
    assert observed_states[1]["hitl_decision"] is None
    assert observed_states[1]["plan_revision_count"] == 0
    assert observed_states[1]["plan_revision_feedback"] is None
    assert [
        message["content"] for message in snapshot.values["messages"] if message["role"] == "user"
    ] == [
        "Request A",
        "Request B",
    ]
    assert snapshot.values["user_request"] == "Request B"
    assert snapshot.values["execution_job_id"] == "job-B"
    assert snapshot.values["decision"] == "Request B executed"


def test_worker_bounds_failed_delivery_attempts() -> None:
    class FailingGraph:
        async def aget_state(self, _config):
            return SimpleNamespace(values={}, interrupts=(), next=())

        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("transient graph failure")

    queue = FakeQueue()
    job_id = "0dd55c94-c67a-407b-b8cc-59b73e6d60ad"
    thread_id = "c05de08e-ea2b-4372-a806-7a306191ef45"
    job = {
        "job_id": job_id,
        "thread_id": thread_id,
        "operation": "chat",
        "payload": json.dumps({"message": "hello", "model_alias": "travel_general"}),
        "status": "queued",
    }
    queue.jobs[job_id] = job
    settings = SimpleNamespace(
        redis_claim_idle_ms=300_000,
        redis_heartbeat_seconds=30.0,
        job_max_delivery_attempts=2,
    )
    worker = StreamWorker(queue, FailingGraph(), FakePool(), settings)
    fields = {**job, "stream_id": "1-0"}

    async def exercise():
        queue.delivery_count = 1
        await worker._handle("worker-1", "1-0", fields)
        first_status = job["status"]
        queue.delivery_count = 2
        await worker._handle("worker-1", "1-0", fields)
        return first_status, job["status"]

    first_status, final_status = asyncio.run(exercise())
    assert first_status == "queued"
    assert final_status == "failed"
    assert queue.acks == ["1-0"]


@pytest.mark.parametrize(
    ("routing_message", "expected_message"),
    [
        ("Supervisor returned malformed JSON", "Supervisor returned malformed JSON"),
        ("private-user@example.com", "WorkflowRoutingError message redacted"),
    ],
)
def test_worker_logs_routing_error_message_and_traceback_safely(
    routing_message: str, expected_message: str
) -> None:
    queue = FakeQueue()
    job_id = "08db58e4-4388-4973-bce2-02301439ce4a"
    thread_id = "fbc7d47e-3db9-4a79-bc7b-65cd436ae25c"
    job = {
        "job_id": job_id,
        "thread_id": thread_id,
        "operation": "chat",
        "payload": json.dumps(
            {"message": "private request content", "model_alias": "travel_local"}
        ),
        "status": "queued",
    }
    queue.jobs[job_id] = job
    worker = StreamWorker(
        queue,
        object(),
        FakePool(),
        SimpleNamespace(
            redis_claim_idle_ms=300_000,
            redis_heartbeat_seconds=30.0,
            job_max_delivery_attempts=5,
        ),
    )

    async def fail_execution(_job):
        raise WorkflowRoutingError(routing_message)

    worker._execute = fail_execution
    fields = {**job, "stream_id": "1-0"}

    async def exercise():
        with capture_logs() as logs:
            await worker._handle("worker-1", "1-0", fields)
        return logs

    logs = asyncio.run(exercise())
    event = next(log for log in logs if log["event"] == "job_execution_failed")
    assert event["error_type"] == "WorkflowRoutingError"
    assert event["error_message"] == expected_message
    assert "fail_execution" in event["traceback"]
    assert "private request content" not in str(event)
    if routing_message == "private-user@example.com":
        assert routing_message not in str(event)


def test_worker_stops_polling_after_shutdown_signal() -> None:
    class StopQueue(FakeQueue):
        async def initialize(self):
            return None

        async def claim_one(self, *_args, **_kwargs):
            return []

        async def read_one(self, *_args, **_kwargs):
            worker.stop()
            return []

    queue = StopQueue()
    worker = StreamWorker(
        queue,
        object(),
        FakePool(),
        SimpleNamespace(
            redis_claim_idle_ms=300_000,
            redis_heartbeat_seconds=30.0,
            job_max_delivery_attempts=5,
        ),
    )
    asyncio.run(worker.run("worker-1"))
    assert worker.stopping.is_set()

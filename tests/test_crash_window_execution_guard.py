import pytest
from unittest import mock
from contextlib import asynccontextmanager
from app.core.advisory_lock import thread_advisory_lock

@asynccontextmanager
async def _dummy_lock(*_, **__):
    yield None

# Replace advisory lock with dummy for tests
thread_advisory_lock = _dummy_lock

# Minimal in-memory JobQueue – same semantics as previous version
class DummyJobQueue:
    def __init__(self):
        self.jobs = {}
        self.acknowledged = False
        self.status_set = []  # (job_id, status, error)

    async def initialize(self):
        pass

    async def get(self, job_id: str):
        # Return the full job dict if stored, otherwise minimal info.
        job = self.jobs.get(job_id)
        if isinstance(job, dict):
            return job
        return {
            "job_id": job_id,
            "status": self.jobs.get(job_id, "queued"),
            "thread_id": "dummy-thread",
        }

    async def set_status(self, job_id: str, status: str, *, error: str = ""):
        # Preserve job dict; record status without overwriting job entry
        self.status_set.append((job_id, status, error))

    async def acknowledge(self, stream_id: str):
        self.acknowledged = True

    async def deliveries(self, stream_id: str):
        return 0

    async def claim_one(self, consumer: str, *, min_idle_ms: int):
        return []

    async def read_one(self, consumer: str, *, block_ms: int = 1000):
        return []

    async def heartbeat(self, consumer: str, stream_id: str):
        pass

    async def close(self):
        pass

# Helper to invoke the private ``_handle`` method
async def run_handle(worker, consumer, stream_id, fields):
    await worker._handle(consumer, stream_id, fields)

@pytest.mark.asyncio
async def test_crash_window_execution_guard():
    """Guard ``execution_job_id`` should stop a second worker from executing the same job when the first worker crashes after ``_execute`` but before the ``succeeded`` status is persisted."""
    # Shared mutable snapshot – both workers will see the same checkpoint.
    class DummySnapshot:
        def __init__(self):
            self.values = {}
            # truthy -> there is pending work
            self.next = ("placeholder",)

    snapshot = DummySnapshot()

    class DummyPool:
        class DummyConnection:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                return False
            async def execute(self, *args, **kwargs):
                return None

        def connection(self):
            return self.DummyConnection()

    # Mock graph – only aget_state and ainvoke are used inside ``_execute``.
    dummy_graph = mock.AsyncMock()
    dummy_graph.aget_state.return_value = snapshot

    # Settings mock (only the fields used by the worker are needed)
    settings = mock.Mock()
    settings.redis_claim_idle_ms = 0
    settings.job_max_delivery_attempts = 5

    # JobQueue – make set_status('succeeded') raise to simulate a crash.
    job_queue = DummyJobQueue()
    original_set_status = job_queue.set_status

    async def failing_set_status(job_id: str, status: str, *, error: str = ""):
        if status == "succeeded":
            raise RuntimeError("simulated crash before status write")
        await original_set_status(job_id, status, error=error)

    job_queue.set_status = failing_set_status  # type: ignore[assignment]

    from app.worker import StreamWorker
    worker1 = StreamWorker(jobs=job_queue, graph=dummy_graph, pool=DummyPool(), settings=settings)

    # Minimal valid payload for the ``chat`` operation
    snapshot.interrupts = ()

    fields = {
        "job_id": "test-job-crash",
        "thread_id": "t-crash",
        "operation": "chat",
        "payload": "{\"model_alias\": \"test-model\", \"message\": \"hello\"}",
    }
    # Store the full job dict in the dummy queue so that jobs.get returns it
    job_queue.jobs[fields["job_id"]] = fields

    # Mock graph invoke writes checkpoint (simulates graph execution)
    async def invoke_side_effect(*_, **__):
        snapshot.values["execution_job_id"] = fields["job_id"]
        snapshot.next = ()
        return None

    dummy_graph.ainvoke.side_effect = invoke_side_effect

    await run_handle(worker1, "consumer1", "stream-1", fields)

    # Checkpoint written by mock graph during invoke_side_effect
    assert snapshot.values.get("execution_job_id") == "test-job-crash"
    assert ("test-job-crash", "running", "") in job_queue.status_set
    assert all(s != "succeeded" for _, s, _ in job_queue.status_set)
    assert dummy_graph.ainvoke.call_count == 1

    # Second worker – should see the checkpoint and skip execution.
    worker2 = StreamWorker(jobs=job_queue, graph=dummy_graph, pool=DummyPool(), settings=settings)
    dummy_graph.ainvoke.reset_mock()

    await run_handle(worker2, "consumer2", "stream-1", fields)

    assert dummy_graph.ainvoke.call_count == 0
    # Two running status sets (one per worker) expected
    assert job_queue.status_set.count(("test-job-crash", "running", "")) == 2


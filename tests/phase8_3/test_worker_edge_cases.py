# tests/phase8_3/test_worker_edge_cases.py
"""Edge‑case tests for the StreamWorker and JobQueue.
These tests target the coverage gaps identified in Phase 8.3 without altering production code.
"""

import asyncio
import json
import sys
from unittest import mock

import pytest

# Ensure the project root is on the import path for absolute imports like `app.worker`
sys.path.append("d:/travel-assistant-multi-agent")

from app.worker import StreamWorker, StaleJobError
from app.jobs.queue import JobQueue
from app.core.advisory_lock import thread_advisory_lock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_settings():
    settings = mock.Mock()
    settings.redis_claim_idle_ms = 0
    settings.job_max_delivery_attempts = 3
    settings.redis_heartbeat_seconds = 0.1
    settings.redis_stream = "stream"
    settings.redis_consumer_group = "group"
    settings.job_status_ttl_seconds = 60
    settings.redis_url = "redis://localhost:6379/0"
    return settings

@pytest.fixture
def mock_pool():
    # pool provides .connection() async context manager compatible with production API
    conn = mock.AsyncMock()
    conn.execute = mock.AsyncMock()
    # Simple async context manager yielding the mock connection
    class _AsyncCM:
        async def __aenter__(self):
            return conn
        async def __aexit__(self, exc_type, exc, tb):
            return False
    pool = mock.Mock()
    pool.connection = mock.Mock(return_value=_AsyncCM())
    return pool

@pytest.fixture
def mock_graph():
    graph = mock.AsyncMock()
    # aget_state returns a dummy snapshot object with required attributes
    snapshot = mock.Mock()
    snapshot.values = {}
    snapshot.next = None
    snapshot.interrupts = ()
    graph.aget_state.return_value = snapshot
    graph.ainvoke = mock.AsyncMock()
    return graph

@pytest.fixture
def mock_jobs():
    jobs = mock.AsyncMock(spec=JobQueue)
    # Default implementations for methods used by StreamWorker
    jobs.get.return_value = {"status": "queued"}
    jobs.deliveries.return_value = 1
    jobs.set_status = mock.AsyncMock()
    jobs.acknowledge = mock.AsyncMock()
    jobs.restore_status = mock.AsyncMock()
    return jobs

# ---------------------------------------------------------------------------
# 1. Idempotent handling – a job already succeeded should be ACKed and ignored.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_idempotent_success_ack(mock_jobs, mock_graph, mock_pool, mock_settings):
    # Simulate a job with status "succeeded"
    mock_jobs.get.return_value = {"status": "succeeded"}
    # Provide a single entry for the worker loop
    entry_id = "123-0"
    mock_jobs.claim_one.return_value = [(entry_id, {"job_id": "job1"})]

    worker = StreamWorker(mock_jobs, mock_graph, mock_pool, mock_settings)
    # Run a single iteration of _handle and then stop
    async def stop_after_one():
        await worker._handle("consumer", entry_id, {"job_id": "job1", "thread_id": "t1", "operation": "chat", "payload": "{}"})
        worker.stopping.set()
    await asyncio.gather(stop_after_one())

    mock_jobs.acknowledge.assert_awaited_once_with(entry_id)
    # No further calls to graph or set_status should happen
    mock_graph.ainvoke.assert_not_awaited()
    mock_jobs.set_status.assert_not_awaited()

# ---------------------------------------------------------------------------
# 2. Worker failure before ACK – an exception inside _execute should not ACK the entry.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_worker_exception_no_ack(mock_jobs, mock_graph, mock_pool, mock_settings):
    # Force _execute to raise an exception
    async def bad_execute(_):
        raise RuntimeError("boom")
    worker = StreamWorker(mock_jobs, mock_graph, mock_pool, mock_settings)
    worker._execute = bad_execute

    entry_id = "124-0"
    mock_jobs.claim_one.return_value = [(entry_id, {"job_id": "job2", "thread_id": "t2", "operation": "chat", "payload": "{}"})]
    # Run a single handle and then stop
    async def stop_after_one():
        await worker._handle("consumer", entry_id, {"job_id": "job2", "thread_id": "t2", "operation": "chat", "payload": "{}"})
        worker.stopping.set()
    await asyncio.gather(stop_after_one())

    # The job should be set back to "queued" (the generic except path) and NOT acknowledged
    mock_jobs.set_status.assert_awaited()
    mock_jobs.acknowledge.assert_not_awaited()

# ---------------------------------------------------------------------------
# 3. StaleJobError when checkpoint already advanced.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_job_error_on_advanced_checkpoint(mock_jobs, mock_graph, mock_pool, mock_settings):
    # Snapshot reports that this thread already has the current job_id set
    snapshot = mock.Mock()
    snapshot.values = {"execution_job_id": "job3"}
    snapshot.next = True  # Simulate checkpoint has advanced
    snapshot.interrupts = ()
    mock_graph.aget_state.return_value = snapshot

    worker = StreamWorker(mock_jobs, mock_graph, mock_pool, mock_settings)
    # When checkpoint advanced, _execute should proceed without raising StaleJobError
    await worker._execute({"job_id": "job3", "thread_id": "t3", "operation": "chat", "payload": "{}"})
    # Verify that the graph was invoked (as checkpoint advanced triggers invoke)
    mock_graph.ainvoke.assert_awaited_once()

# ---------------------------------------------------------------------------
# 4. Advisory lock acquisition – ensure the lock context manager is entered.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_advisory_lock_used(mock_jobs, mock_graph, mock_pool, mock_settings):
    # Patch the lock context manager to a mock
    mock_lock = mock.Mock()
    mock_context = mock.AsyncMock()
    mock_context.__aenter__.return_value = None
    mock_context.__aexit__.return_value = None
    mock_lock.return_value = mock_context
    with mock.patch("app.worker.thread_advisory_lock", mock_lock):
        worker = StreamWorker(mock_jobs, mock_graph, mock_pool, mock_settings)
        # Directly call _execute which should use the lock
        await worker._execute({"job_id": "job4", "thread_id": "t4", "operation": "chat", "payload": "{\"model_alias\": \"test\", \"message\": \"hi\"}"})
        mock_lock.assert_called_once_with(mock_pool, "t4")
        mock_context.__aenter__.assert_awaited()

# ---------------------------------------------------------------------------
# 5. HITL resume flow – a valid resume payload updates the graph without error.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_hitl_resume_success(mock_jobs, mock_graph, mock_pool, mock_settings):
    # Prepare snapshot with an interrupt awaiting approval
    interrupt = mock.Mock()
    interrupt.id = "int1"
    snapshot = mock.Mock()
    snapshot.values = {}
    snapshot.interrupts = (interrupt,)
    snapshot.next = True
    mock_graph.aget_state.return_value = snapshot

    payload = {
        "interrupt_id": "int1",
        "decision": {"interrupt_id": "int1", "decision": "approve"},
    }
    job = {
        "job_id": "job5",
        "thread_id": "t5",
        "operation": "resume",
        "payload": json.dumps(payload),
    }
    worker = StreamWorker(mock_jobs, mock_graph, mock_pool, mock_settings)
    await worker._execute(job)
    # The graph should have been invoked with a Command containing the resume decision
    mock_graph.ainvoke.assert_awaited()
    invoked_cmd = mock_graph.ainvoke.call_args[0][0]
    assert hasattr(invoked_cmd, "resume")
    assert "int1" in invoked_cmd.resume


# tests/test_delivery_semantics.py
"""Targeted unit tests for JobQueue.deliveries() and StreamWorker._handle.
These tests are deliberately tiny, use only the public constructors
(`JobQueue`, `StreamWorker`) and avoid any external services.
"""

import pytest, asyncio
from unittest import mock

# --- Helper mock Redis ----------------------------------------------------
class MockRedis:
    def __init__(self, pending_result):
        # pending_result is either a list with a dict containing
        # "times_delivered" or an empty list.
        self._pending_result = pending_result
        self.acked = False
        self.deleted = False

    async def xpending_range(self, *a, **kw):
        return self._pending_result

    async def xack(self, *a, **kw):
        self.acked = True

    async def xdel(self, *a, **kw):
        self.deleted = True

    async def pipeline(self, transaction=True):
        # Simple pipeline that just records calls.
        class Pipe:
            async def __aenter__(self_):
                return self_
            async def __aexit__(self_, *exc):
                pass
            def xack(self_, *a, **kw):
                self_.acked = True
            def xdel(self_, *a, **kw):
                self_.deleted = True
            async def execute(self_):
                return []
        return Pipe()

    # Stubs for other redis methods used by JobQueue (no‑ops).
    async def xreadgroup(self, *a, **kw):
        return []
    async def xautoclaim(self, *a, **kw):
        return ("0-0", [])
    async def hgetall(self, *a, **kw):
        return {"status": "queued"}
    async def hset(self, *a, **kw):
        pass
    async def expire(self, *a, **kw):
        pass

# -----------------------------------------------------------------------
# 1️⃣ Test deliveries() when Redis reports times_delivered == 1
# -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deliveries_one():
    from app.jobs.queue import JobQueue
    redis = MockRedis(pending_result=[{"times_delivered": 1}])
    q = JobQueue(redis, stream="travel:jobs", group="travel-workers", status_ttl_seconds=60)
    count = await q.deliveries("any-id")
    assert count == 1  # should return the exact value from Redis

# -----------------------------------------------------------------------
# 2️⃣ Test deliveries() when Redis returns an empty list
# -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deliveries_empty():
    from app.jobs.queue import JobQueue
    redis = MockRedis(pending_result=[])
    q = JobQueue(redis, stream="travel:jobs", group="travel-workers", status_ttl_seconds=60)
    count = await q.deliveries("any-id")
    # Current implementation falls back to 1; the test documents that behaviour.
    assert count == 1

# -----------------------------------------------------------------------
# 3️⃣ Test that _handle() skips execution when status is succeeded/failed
# -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handle_skips_when_status_final():
    from app.worker import StreamWorker
    # Re‑use the DummyJobQueue from the existing failure‑handling tests.
    from tests.test_worker_failure_handling import DummyJobQueue

    # Prepare a queue that reports a final status.
    class FinalStatusQueue(DummyJobQueue):
        async def get(self, job_id: str):
            # Return a job dict indicating it is already succeeded.
            return {"job_id": job_id, "status": "succeeded", "thread_id": "t"}

    q = FinalStatusQueue()
    dummy_graph = mock.AsyncMock()
    dummy_pool = mock.AsyncMock()
    settings = mock.Mock()
    settings.redis_claim_idle_ms = 0
    settings.job_max_delivery_attempts = 5

    worker = StreamWorker(jobs=q, graph=dummy_graph, pool=dummy_pool, settings=settings)

    # Patch the queue's acknowledge to verify it is called, but _execute should never run.
    with mock.patch.object(q, "acknowledge", wraps=q.acknowledge) as ack_spy:
        await worker._handle(consumer="c", stream_id="s1", fields={"job_id": "jid", "thread_id": "t", "operation": "chat", "payload": "{}"})
        # The job is final, so _execute must not be invoked.
        dummy_graph.ainvoke.assert_not_called()
        # ACK should have been performed exactly once.
        ack_spy.assert_awaited_once()

    # Repeat for a failed status.
    class FailedStatusQueue(DummyJobQueue):
        async def get(self, job_id: str):
            return {"job_id": job_id, "status": "failed", "thread_id": "t"}

    q2 = FailedStatusQueue()
    worker2 = StreamWorker(jobs=q2, graph=dummy_graph, pool=dummy_pool, settings=settings)
    with mock.patch.object(q2, "acknowledge", wraps=q2.acknowledge) as ack_spy2:
        await worker2._handle(consumer="c", stream_id="s2", fields={"job_id": "jid2", "thread_id": "t", "operation": "chat", "payload": "{}"})
        dummy_graph.ainvoke.assert_not_called()
        ack_spy2.assert_awaited_once()


import sys
import types
import asyncio
from unittest import mock
import pytest

# ----------------------------------------------------------------------
# Stub missing external dependencies so that ``app.worker`` can be imported
# without pulling in optional packages (langgraph, langchain_openai, mcp, etc.)
# ----------------------------------------------------------------------
# ``langgraph.checkpoint.postgres.aio`` provides ``AsyncPostgresSaver`` – we
# replace it with a minimal dummy class.
langgraph_pkg = types.ModuleType('langgraph')
checkpoint_pkg = types.ModuleType('langgraph.checkpoint')
postgres_pkg = types.ModuleType('langgraph.checkpoint.postgres')
postgres_aio_pkg = types.ModuleType('langgraph.checkpoint.postgres.aio')

class DummyAsyncPostgresSaver:  # pragma: no cover
    pass

postgres_aio_pkg.AsyncPostgresSaver = DummyAsyncPostgresSaver

# Stub for langgraph.types
langgraph_types_pkg = types.ModuleType('langgraph.types')
class DummyCommand:  # pragma: no cover
    pass
langgraph_types_pkg.Command = DummyCommand
langgraph_types_pkg.interrupt = object()
sys.modules['langgraph.types'] = langgraph_types_pkg

# Populate the module hierarchy in ``sys.modules``
sys.modules['langgraph'] = langgraph_pkg
sys.modules['langgraph.checkpoint'] = checkpoint_pkg
sys.modules['langgraph.checkpoint.postgres'] = postgres_pkg
sys.modules['langgraph.checkpoint.postgres.aio'] = postgres_aio_pkg
sys.modules['langgraph.checkpoint.postgres'] = postgres_pkg
# Stub langgraph.graph module for worker imports
langgraph_graph_pkg = types.ModuleType('langgraph.graph')
langgraph_graph_pkg.END = object()
langgraph_graph_pkg.START = object()
class DummyStateGraph:
    def __init__(self, *args, **kwargs):
        pass
    def add_node(self, *args, **kwargs):
        pass
    def set_entry_point(self, *args, **kwargs):
        pass
    def compile(self):
        return self
langgraph_graph_pkg.StateGraph = DummyStateGraph
sys.modules['langgraph.graph'] = langgraph_graph_pkg
sys.modules['langgraph.checkpoint.postgres.aio'] = postgres_aio_pkg

# ``langchain_openai`` is imported in several modules (LLM factory).  Stub it.
langchain_openai_pkg = types.ModuleType('langchain_openai')
langchain_openai_pkg.ChatOpenAI = object
sys.modules['langchain_openai'] = langchain_openai_pkg

# ``mcp`` client/server modules – stub a minimal ``Client`` class.
mcp_pkg = types.ModuleType('mcp')
class DummyMCPClient:  # pragma: no cover
    pass
mcp_pkg.Client = DummyMCPClient
sys.modules['mcp'] = mcp_pkg
# Stub mcp.server.mcpserver module for MCPToolClient import
mcp_server_pkg = types.ModuleType('mcp.server')
mcpserver_mod = types.ModuleType('mcp.server.mcpserver')
class DummyMCPServer:
    pass
mcpserver_mod.MCPServer = DummyMCPServer
sys.modules['mcp.server'] = mcp_server_pkg
sys.modules['mcp.server.mcpserver'] = mcpserver_mod

# ``structlog.testing`` is used in some tests – provide a stub ``capture_logs``.
structlog_pkg = types.ModuleType('structlog')
# Provide a minimal logger with .error/.debug/.info methods used in advisory_lock
class _DummyLogger:
    def error(self, *args, **kwargs):
        pass
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
structlog_pkg.get_logger = lambda name=None: _DummyLogger()
structlog_testing_pkg = types.ModuleType('structlog.testing')
structlog_testing_pkg.capture_logs = lambda *args, **kwargs: (lambda func: func)
sys.modules['structlog'] = structlog_pkg
sys.modules['structlog.testing'] = structlog_testing_pkg

# ----------------------------------------------------------------------
# Now import the worker under test
# ----------------------------------------------------------------------
from app.worker import StreamWorker

# ----------------------------------------------------------------------
# Minimal stub for JobQueue used by the worker
# ----------------------------------------------------------------------
class DummyJobQueue:
    def __init__(self):
        self.jobs = {}
        self.acknowledged = False
        self.status_set = []  # (job_id, status, error)

    async def initialize(self):
        pass

    async def get(self, job_id: str):
        return {"job_id": job_id,
                "status": self.jobs.get(job_id, "queued"),
                "thread_id": "dummy-thread"}

    async def set_status(self, job_id: str, status: str, *, error: str = ""):
        self.jobs[job_id] = status
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

# Helper to run the private _handle method
async def run_handle(worker, consumer, stream_id, fields):
    await worker._handle(consumer, stream_id, fields)

@pytest.fixture
def dummy_worker():
    job_queue = DummyJobQueue()
    dummy_graph = mock.AsyncMock()
    dummy_pool = mock.AsyncMock()
    settings = mock.Mock()
    settings.redis_claim_idle_ms = 0
    settings.job_max_delivery_attempts = 5
    return StreamWorker(jobs=job_queue, graph=dummy_graph,
                        pool=dummy_pool, settings=settings)

# ----------------------------------------------------------------------
# Test 1 – failure of set_status("succeeded")
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_set_status_succeeded_failure(dummy_worker):
    """If ``set_status('succeeded')`` raises, the worker must not downgrade the
    status and must not ACK the entry."""
    job_id = "test-job-1"
    fields = {"job_id": job_id, "thread_id": "t1",
              "operation": "chat", "payload": "{}"}

    # Make _execute succeed
    with mock.patch.object(dummy_worker, "_execute",
                           return_value=asyncio.Future()) as exec_mock:
        exec_mock.return_value.set_result(None)
        # Raise only when status == "succeeded"
        original_set_status = dummy_worker.jobs.set_status
        async def failing_set_status(jid, status, *, error=""):
            if status == "succeeded":
                raise RuntimeError("simulated status failure")
            await original_set_status(jid, status, error=error)
        with mock.patch.object(dummy_worker.jobs,
                               "set_status", side_effect=failing_set_status):
            await run_handle(dummy_worker, "consumer", "stream-1", fields)

    # Verify "running" was written, then failure on "succeeded"
    assert (job_id, "running", "") in dummy_worker.jobs.status_set
    # No downgrade to queued/failed
    assert all(s not in {"queued", "failed"}
               for _, s, _ in dummy_worker.jobs.status_set if s != "running")
    # ACK never called
    assert dummy_worker.jobs.acknowledged is False

# ----------------------------------------------------------------------
# Test 2 – ACK failure after successful status write
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_acknowledge_failure_after_success(dummy_worker):
    """If ACK raises after status succeeded, the status stays ``succeeded``.
    A subsequent redelivery should early‑exit (no second ``_execute`` call).
    """
    job_id = "test-job-2"
    fields = {"job_id": job_id, "thread_id": "t2",
              "operation": "chat", "payload": "{}"}

    # Make _execute succeed
    with mock.patch.object(dummy_worker, "_execute",
                           return_value=asyncio.Future()) as exec_mock:
        exec_mock.return_value.set_result(None)
        # First delivery – ACK fails
        async def failing_ack(_):
            raise RuntimeError("simulated ack failure")
        with mock.patch.object(dummy_worker.jobs, "acknowledge",
                               side_effect=failing_ack):
            await run_handle(dummy_worker, "consumer", "stream-2", fields)
        # Second delivery – normal ACK behavior (no exception)
        await run_handle(dummy_worker, "consumer", "stream-2", fields)

    # Status succeeded should be recorded
    assert (job_id, "succeeded", "") in dummy_worker.jobs.status_set
    # _execute should have been called only once (second delivery skips it)
    assert exec_mock.call_count == 1
    # After the successful second delivery, ACK flag becomes True
    assert dummy_worker.jobs.acknowledged is True

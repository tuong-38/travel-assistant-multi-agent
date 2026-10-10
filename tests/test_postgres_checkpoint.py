import sys
import asyncio
from app.core.config import get_settings
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import pytest
from urllib.parse import urlparse

from langchain_core.messages import AIMessage

# Import the runtime that yields (pool, graph) using AsyncPostgresSaver
from app.graph.runtime import graph_runtime

# Monkey‑patch the workflow's model factory to provide a deterministic fake model
from app.graph import workflow as wf


class FakeModel:
    """A minimal async model that returns predefined outputs.

    The graph expects ``await model.ainvoke(messages)`` to return an
    ``AIMessage`` whose ``content`` is a JSON string.  For this test we only
    need the supervisor to return a ``FINISH`` decision.
    """

    def __init__(self, outputs: list[str]):
        self._outputs = iter(outputs)

    async def ainvoke(self, messages):  # pragma: no cover – exercised via graph
        return AIMessage(content=next(self._outputs))


@pytest.mark.asyncio
async def test_async_postgres_checkpoint_persistence(monkeypatch):
    """Verify that a checkpoint saved with ``AsyncPostgresSaver`` survives a
    runtime restart.

    The test uses a dedicated database identified by the ``TEST_DATABASE_URL``
    environment variable.  If the variable is missing or points to a different
    database name, the test is skipped or fails with a clear message.
    """

    # ---------------------------------------------------------------------
    # 1. Resolve and validate the dedicated test database URL.
    # ---------------------------------------------------------------------
    test_url = os.getenv("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL not set – PostgreSQL checkpoint test is skipped")

    parsed = urlparse(test_url)
    # ``parsed.path`` includes a leading slash, e.g. "/travel_assistant_checkpoint_test"
    db_name = parsed.path.lstrip("/")
    expected_name = "travel_assistant_checkpoint_test"
    if db_name != expected_name:
        pytest.fail(
            f"TEST_DATABASE_URL must point to database '{expected_name}', got '{db_name}'"
        )

    # ---------------------------------------------------------------------
    # 2. Prepare environment for the application's Settings.
    # ---------------------------------------------------------------------
    # ``app.core.config.Settings`` reads ``DATABASE_URL`` and ``LITELLM_API_KEY``.
    # We deliberately set ``DATABASE_URL`` from ``TEST_DATABASE_URL`` – the test
    # never touches the regular ``DATABASE_URL`` used by the application.
    monkeypatch.setenv("DATABASE_URL", test_url)
    # ``litellm_api_key`` is required but not used in this test.
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-key")
    get_settings.cache_clear()
    # ---------------------------------------------------------------------
    # 3. Patch the LLM factory to return our deterministic fake model.
    # ---------------------------------------------------------------------
    fake_outputs = [
        '{"next":"FINISH","response":"test final"}'
    ]
    monkeypatch.setattr(wf, "get_chat_model", lambda alias: FakeModel(fake_outputs))

    # ---------------------------------------------------------------------
    # 4. First runtime – write a checkpoint.
    # ---------------------------------------------------------------------
    thread_id = "test-thread-123"
    config = {"configurable": {"thread_id": thread_id, "model_alias": "travel_local"}}

    async with graph_runtime() as (_pool, graph):
        # Minimal user request – the fake model will finish immediately.
        input_data = {
            "messages": [{"role": "user", "content": "ping"}],
            "user_request": "ping",
        }
        result = await graph.ainvoke(input_data, config=config)
        assert result["final_response"] == "test final"

    # ---------------------------------------------------------------------
    # 5. Second runtime – re‑open and verify persisted state.
    # ---------------------------------------------------------------------
    async with graph_runtime() as (_pool2, graph2):
        snapshot = await graph2.aget_state(config)
        # ``snapshot.values`` holds the restored state dictionary.
        assert snapshot.values["final_response"] == "test final"
        # Ensure the graph has no pending interrupts after restoration.
        assert getattr(snapshot, "interrupts", ()) == ()


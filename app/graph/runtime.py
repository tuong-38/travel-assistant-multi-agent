"""Shared PostgreSQL checkpointer lifecycle for API and background worker."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import get_settings
from app.graph.workflow import build_graph

CHECKPOINT_SETUP_LOCK_ID = 746_219_503


@asynccontextmanager
async def graph_runtime() -> AsyncIterator[tuple[AsyncConnectionPool, Any]]:
    settings = get_settings()
    pool = AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    try:
        await pool.open()
        async with pool.connection() as connection:
            await connection.execute("SELECT pg_advisory_lock(%s)", (CHECKPOINT_SETUP_LOCK_ID,))
            try:
                await AsyncPostgresSaver(pool).setup()
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock(%s)", (CHECKPOINT_SETUP_LOCK_ID,)
                )
        yield pool, build_graph(AsyncPostgresSaver(pool))
    finally:
        await pool.close()

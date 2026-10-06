from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.workflow import build_graph

logger = structlog.get_logger(__name__)
CHECKPOINT_SETUP_LOCK_ID = 746_219_503


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    pool = AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    try:
        await pool.open()
        app.state.db_pool = pool
        async with pool.connection() as connection:
            await connection.execute("SELECT pg_advisory_lock(%s)", (CHECKPOINT_SETUP_LOCK_ID,))
            try:
                checkpointer = AsyncPostgresSaver(pool)
                await checkpointer.setup()
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock(%s)", (CHECKPOINT_SETUP_LOCK_ID,)
                )
        app.state.graph = build_graph(checkpointer)
        logger.info("application_started", environment=settings.app_env)
        yield
    finally:
        await pool.close()
        logger.info("application_stopped")


def create_app() -> FastAPI:
    application = FastAPI(title="Travel Assistant API", version="0.1.0", lifespan=lifespan)
    application.include_router(health_router, prefix="/api/v1")
    application.include_router(chat_router, prefix="/api/v1")
    return application


app = create_app()

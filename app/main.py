from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.graph.runtime import graph_runtime
from app.jobs.queue import JobQueue

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    jobs = JobQueue.from_url(
        settings.redis_url,
        stream=settings.redis_stream,
        group=settings.redis_consumer_group,
        status_ttl_seconds=settings.job_status_ttl_seconds,
    )
    try:
        await jobs.initialize()
        app.state.jobs = jobs
        async with graph_runtime() as (pool, graph):
            app.state.db_pool = pool
            app.state.graph = graph
            logger.info("application_started", environment=settings.app_env)
            yield
    finally:
        await jobs.close()
        logger.info("application_stopped")


def create_app() -> FastAPI:
    application = FastAPI(title="Travel Assistant API", version="0.1.0", lifespan=lifespan)
    application.include_router(health_router, prefix="/api/v1")
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(jobs_router, prefix="/api/v1")
    return application


app = create_app()

"""PostgreSQL advisory locks shared by API and worker graph execution."""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AdvisoryLockError(RuntimeError):
    """Raised when a PostgreSQL thread lock cannot be acquired or released safely."""


async def _discard_connection(connection: Any) -> None:
    try:
        await connection.close()
    except BaseException as close_error:
        logger.error(
            "advisory_lock_connection_discard_failed", error_type=type(close_error).__name__
        )


@asynccontextmanager
async def thread_advisory_lock(pool: Any, thread_id: str) -> AsyncIterator[None]:
    if pool is None:
        raise AdvisoryLockError("PostgreSQL thread locking is unavailable")

    async with pool.connection() as connection:
        try:
            await connection.execute(
                "SELECT pg_advisory_lock(hashtextextended(%s, 0))", (thread_id,)
            )
        except BaseException as lock_error:
            await _discard_connection(connection)
            raise AdvisoryLockError("PostgreSQL thread lock acquisition failed") from lock_error
        try:
            yield
        finally:
            original_error = sys.exception()
            try:
                await connection.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (thread_id,)
                )
            except BaseException as unlock_error:
                await _discard_connection(connection)
                logger.error(
                    "thread_advisory_unlock_failed",
                    thread_id=thread_id,
                    error_type=type(unlock_error).__name__,
                )
                if original_error is None:
                    raise AdvisoryLockError(
                        "PostgreSQL thread lock release failed"
                    ) from unlock_error

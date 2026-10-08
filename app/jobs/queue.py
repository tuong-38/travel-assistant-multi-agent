"""Minimal Redis Streams queue for chat and HITL resume jobs."""

import json
from typing import Any, Literal
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import ResponseError

JobOperation = Literal["chat", "resume"]


class JobQueue:
    def __init__(
        self,
        redis: Redis,
        *,
        stream: str,
        group: str,
        status_ttl_seconds: int,
    ) -> None:
        self.redis = redis
        self.stream = stream
        self.group = group
        self.status_ttl_seconds = status_ttl_seconds
        self.claim_cursor = "0-0"

    @classmethod
    def from_url(cls, url: str, *, stream: str, group: str, status_ttl_seconds: int):
        return cls(
            Redis.from_url(url, decode_responses=True),
            stream=stream,
            group=group,
            status_ttl_seconds=status_ttl_seconds,
        )

    async def initialize(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, self.group, id="0-0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"travel:job:{job_id}"

    async def enqueue(
        self,
        *,
        job_id: UUID,
        thread_id: UUID,
        operation: JobOperation,
        payload: dict[str, Any],
    ) -> None:
        fields = {
            "job_id": str(job_id),
            "thread_id": str(thread_id),
            "operation": operation,
            "payload": json.dumps(payload, separators=(",", ":")),
        }
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.xadd(self.stream, fields)
            pipeline.hset(
                self._job_key(str(job_id)),
                mapping={
                    **fields,
                    "status": "queued",
                    "error": "",
                },
            )
            pipeline.expire(self._job_key(str(job_id)), self.status_ttl_seconds)
            result = await pipeline.execute()
        if not result or not result[0]:
            raise RuntimeError("Redis did not return a stream entry ID")

    async def get(self, job_id: str) -> dict[str, str] | None:
        result = await self.redis.hgetall(self._job_key(job_id))
        return result or None

    async def restore_status(self, fields: dict[str, str]) -> None:
        key = self._job_key(fields["job_id"])
        await self.redis.hset(
            key,
            mapping={**fields, "status": "queued", "error": ""},
        )
        await self.redis.expire(key, self.status_ttl_seconds)

    async def set_status(self, job_id: str, status: str, *, error: str = "") -> None:
        key = self._job_key(job_id)
        await self.redis.hset(key, mapping={"status": status, "error": error[:500]})
        await self.redis.expire(key, self.status_ttl_seconds)

    async def read_one(
        self, consumer: str, *, block_ms: int = 1000
    ) -> list[tuple[str, dict[str, str]]]:
        entries = await self.redis.xreadgroup(
            self.group,
            consumer,
            {self.stream: ">"},
            count=1,
            block=block_ms,
        )
        if not entries:
            return []
        return entries[0][1]

    async def claim_one(
        self, consumer: str, *, min_idle_ms: int
    ) -> list[tuple[str, dict[str, str]]]:
        result = await self.redis.xautoclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=min_idle_ms,
            start_id=self.claim_cursor,
            count=1,
        )
        self.claim_cursor = result[0] if result else "0-0"
        return result[1]

    async def deliveries(self, stream_id: str) -> int:
        pending = await self.redis.xpending_range(
            self.stream, self.group, min=stream_id, max=stream_id, count=1
        )
        return int(pending[0]["times_delivered"]) if pending else 1

    async def heartbeat(self, consumer: str, stream_id: str) -> None:
        await self.redis.xclaim(
            self.stream,
            self.group,
            consumer,
            min_idle_time=0,
            message_ids=[stream_id],
            idle=0,
            justid=True,
        )

    async def acknowledge(self, stream_id: str) -> None:
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.xack(self.stream, self.group, stream_id)
            pipeline.xdel(self.stream, stream_id)
            await pipeline.execute()

    async def close(self) -> None:
        await self.redis.aclose()

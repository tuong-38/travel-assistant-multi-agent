"""Redis Streams consumer that runs checkpointed travel graph jobs."""

import asyncio
import json
import signal
import traceback
from typing import Any

import structlog
from langgraph.types import Command

from app.core.advisory_lock import thread_advisory_lock
from app.core.config import get_settings
from app.graph.hitl import MAX_PLAN_REVISIONS, ResumeDecision
from app.graph.runtime import graph_runtime
from app.graph.state import new_chat_execution_input
from app.graph.workflow import WorkflowRoutingError
from app.jobs.queue import JobQueue

logger = structlog.get_logger(__name__)
_SAFE_ROUTING_ERROR_MESSAGES = {
    "Supervisor returned a non-text decision",
    "Supervisor returned malformed JSON",
    "Supervisor decision has an invalid shape",
    "Supervisor proposed a route outside the allowlist",
    "Supervisor response must be text",
    "Supervisor must provide a response when finishing",
    "Supervisor attempted to repeat a completed specialist",
    "Supervisor selected itinerary before a travel plan exists",
    "Supervisor must finish after itinerary generation",
    "Supervisor selected itinerary before plan approval",
    "Planning requires destination information, but destination lookup is complete",
    "Invalid specialist node selection",
}


class StaleJobError(ValueError):
    """The checkpoint no longer permits the queued operation."""


class StreamWorker:
    def __init__(self, jobs: JobQueue, graph: Any, pool: Any, settings: Any) -> None:
        self.jobs = jobs
        self.graph = graph
        self.pool = pool
        self.settings = settings
        self.stopping = asyncio.Event()

    def stop(self) -> None:
        self.stopping.set()

    async def run(self, consumer: str) -> None:
        await self.jobs.initialize()
        while not self.stopping.is_set():
            entries = await self.jobs.claim_one(
                consumer, min_idle_ms=self.settings.redis_claim_idle_ms
            )
            if not entries:
                entries = await self.jobs.read_one(consumer, block_ms=1000)
            for stream_id, fields in entries:
                if self.stopping.is_set():
                    return
                await self._handle(consumer, stream_id, fields)

    async def _heartbeat(self, consumer: str, stream_id: str) -> None:
        while True:
            await asyncio.sleep(self.settings.redis_heartbeat_seconds)
            await self.jobs.heartbeat(consumer, stream_id)

    async def _handle(self, consumer: str, stream_id: str, fields: dict[str, str]) -> None:
        job_id = fields.get("job_id", "")
        try:
            job = await self.jobs.get(job_id)
            if job is None:
                await self.jobs.restore_status(fields)
                job = await self.jobs.get(job_id)
            if job is None:
                raise RuntimeError("Unable to restore job status metadata")
            if job.get("status") in {"succeeded", "failed"}:
                await self.jobs.acknowledge(stream_id)
                return

            deliveries = await self.jobs.deliveries(stream_id)
            if deliveries > self.settings.job_max_delivery_attempts:
                await self.jobs.set_status(
                    job_id,
                    "failed",
                    error=(
                        "Maximum delivery attempts exceeded; checkpoint may require operator review"
                    ),
                )
                await self.jobs.acknowledge(stream_id)
                return

            await self.jobs.set_status(job_id, "running")
            heartbeat = asyncio.create_task(self._heartbeat(consumer, stream_id))
            try:
                await self._execute(job)
                await self.jobs.set_status(job_id, "succeeded")
                await self.jobs.acknowledge(stream_id)
            except StaleJobError as exc:
                await self.jobs.set_status(job_id, "failed", error=str(exc))
                await self.jobs.acknowledge(stream_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_fields = {
                    "job_id": job_id,
                    "thread_id": fields.get("thread_id"),
                    "delivery": deliveries,
                    "error_type": type(exc).__name__,
                }
                if isinstance(exc, WorkflowRoutingError):
                    message = str(exc)
                    log_fields["error_message"] = (
                        message
                        if message in _SAFE_ROUTING_ERROR_MESSAGES
                        else "WorkflowRoutingError message redacted"
                    )
                    log_fields["traceback"] = "".join(traceback.format_tb(exc.__traceback__))
                logger.error("job_execution_failed", **log_fields)
                if deliveries >= self.settings.job_max_delivery_attempts:
                    await self.jobs.set_status(
                        job_id,
                        "failed",
                        error="Job failed after the maximum delivery attempts",
                    )
                    await self.jobs.acknowledge(stream_id)
                else:
                    await self.jobs.set_status(job_id, "queued", error="")
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "job_delivery_failed",
                job_id=job_id,
                error_type=type(exc).__name__,
            )

    async def _execute(self, job: dict[str, str]) -> None:
        thread_id = job["thread_id"]
        job_id = job["job_id"]
        operation = job["operation"]
        payload = json.loads(job["payload"])
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        async with thread_advisory_lock(self.pool, thread_id):
            snapshot = await self.graph.aget_state(config)
            values = getattr(snapshot, "values", {}) or {}
            if values.get("execution_job_id") == job_id:
                if not getattr(snapshot, "next", ()):
                    return
                alias = values.get("model_alias") or payload.get("model_alias")
                if alias:
                    config["configurable"]["model_alias"] = alias
                await self.graph.ainvoke(None, config=config)
                return

            if operation == "chat":
                if getattr(snapshot, "interrupts", ()):
                    raise StaleJobError("Thread is awaiting approval; queued chat was not run")
                alias = payload["model_alias"]
                config["configurable"]["model_alias"] = alias
                await self.graph.ainvoke(
                    new_chat_execution_input(payload["message"], alias, job_id),
                    config=config,
                )
                return

            if operation != "resume":
                raise StaleJobError("Unsupported job operation")
            interrupts = getattr(snapshot, "interrupts", ())
            if not interrupts or interrupts[0].id != payload.get("interrupt_id"):
                raise StaleJobError("Approval interrupt is stale")
            decision = ResumeDecision.model_validate(payload["decision"])
            if (
                decision.decision == "modify"
                and values.get("plan_revision_count", 0) >= MAX_PLAN_REVISIONS
            ):
                raise StaleJobError("Maximum plan revisions exceeded")
            alias = values.get("model_alias")
            if alias:
                config["configurable"]["model_alias"] = alias
            await self.graph.ainvoke(
                Command(
                    resume={interrupts[0].id: decision.model_dump(exclude_unset=True)},
                    update={"execution_job_id": job_id},
                ),
                config=config,
            )


async def run_worker() -> None:
    settings = get_settings()
    jobs = JobQueue.from_url(
        settings.redis_url,
        stream=settings.redis_stream,
        group=settings.redis_consumer_group,
        status_ttl_seconds=settings.job_status_ttl_seconds,
    )
    try:
        await jobs.initialize()
        async with graph_runtime() as (pool, graph):
            worker = StreamWorker(jobs, graph, pool, settings)
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, worker.stop)
                except NotImplementedError:
                    pass
            task = asyncio.create_task(worker.run(consumer=f"worker-{id(worker)}"))
            stop_waiter = asyncio.create_task(worker.stopping.wait())
            try:
                done, _ = await asyncio.wait(
                    {task, stop_waiter}, return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    await task
                else:
                    try:
                        await asyncio.wait_for(task, timeout=settings.worker_shutdown_grace_seconds)
                    except TimeoutError:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            finally:
                stop_waiter.cancel()
                try:
                    await stop_waiter
                except asyncio.CancelledError:
                    pass
    finally:
        await jobs.close()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()

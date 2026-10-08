import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.advisory_lock import AdvisoryLockError, thread_advisory_lock
from app.core.config import get_settings
from app.graph.hitl import MAX_PLAN_REVISIONS, ResumeDecision
from app.graph.state import new_chat_execution_input

_thread_locks: dict[str, asyncio.Lock] = {}

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(default=None, min_length=1)
    thread_id: UUID | None = None


class ChatResponse(BaseModel):
    thread_id: UUID
    reply: str
    status: Literal["interrupted", "rejected"] | None = None
    pending_approval: dict[str, Any] | None = None


def _active_interrupt(snapshot: Any) -> Any | None:
    interrupts = getattr(snapshot, "interrupts", ())
    return interrupts[0] if len(interrupts) == 1 else None


def _has_checkpoint(snapshot: Any) -> bool:
    return bool(getattr(snapshot, "values", None) or getattr(snapshot, "next", ()))


@asynccontextmanager
async def _thread_resume_lock(request: Request, thread_id: str):
    """Serialize graph operations with a PostgreSQL advisory lock in production."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        if getattr(request.app.state, "testing", False) is True:
            lock = _thread_locks.setdefault(thread_id, asyncio.Lock())
            async with lock:
                yield
            return
        raise HTTPException(status_code=503, detail="Resume locking is unavailable")

    try:
        async with thread_advisory_lock(pool, thread_id):
            yield
    except AdvisoryLockError as exc:
        raise HTTPException(
            status_code=503, detail="PostgreSQL thread locking is unavailable"
        ) from exc


@router.post("", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    thread_id = payload.thread_id or uuid4()
    graph = request.app.state.graph
    model_alias = payload.model or get_settings().default_model
    config = {"configurable": {"thread_id": str(thread_id), "model_alias": model_alias}}
    try:
        async with _thread_resume_lock(request, str(thread_id)):
            if hasattr(graph, "aget_state"):
                snapshot = await graph.aget_state(config)
                if _active_interrupt(snapshot) is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="This thread is awaiting approval; use the resume endpoint",
                    )
            result = await graph.ainvoke(
                new_chat_execution_input(payload.message, model_alias),
                config=config,
            )
            snapshot = await graph.aget_state(config) if hasattr(graph, "aget_state") else None
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "chat_request_failed",
            thread_id=str(thread_id),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Chat workflow is unavailable") from exc

    active_interrupt = _active_interrupt(snapshot) if snapshot is not None else None
    if active_interrupt is not None:
        values = snapshot.values
        return ChatResponse(
            thread_id=thread_id,
            reply="Your travel plan is ready for review.",
            status="interrupted",
            pending_approval={
                "interrupt_id": active_interrupt.id,
                "travel_plan": values.get("travel_plan"),
                "plan_revision_count": values.get("plan_revision_count", 0),
                "max_plan_revisions": MAX_PLAN_REVISIONS,
            },
        )
    reply: str = result.get("final_response") or result["messages"][-1]["content"]
    return ChatResponse(thread_id=thread_id, reply=reply)


@router.post("/{thread_id}/resume", response_model=ChatResponse, response_model_exclude_none=True)
async def resume_chat(thread_id: UUID, payload: dict[str, Any], request: Request) -> ChatResponse:
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": str(thread_id)}}
    try:
        async with _thread_resume_lock(request, str(thread_id)):
            snapshot = await graph.aget_state(config)
            if not _has_checkpoint(snapshot):
                raise HTTPException(status_code=404, detail="Conversation thread was not found")
            active_interrupt = _active_interrupt(snapshot)
            if active_interrupt is None:
                raise HTTPException(status_code=409, detail="Thread has no pending approval")
            if payload.get("interrupt_id") != active_interrupt.id:
                raise HTTPException(status_code=409, detail="Approval interrupt is stale")
            try:
                decision = ResumeDecision.model_validate(payload)
            except Exception as exc:
                raise HTTPException(status_code=422, detail="Invalid approval decision") from exc
            values = snapshot.values
            if (
                decision.decision == "modify"
                and values.get("plan_revision_count", 0) >= MAX_PLAN_REVISIONS
            ):
                raise HTTPException(status_code=409, detail="Maximum plan revisions exceeded")
            model_alias = values.get("model_alias")
            if model_alias:
                config["configurable"]["model_alias"] = model_alias
            resume_value = decision.model_dump(exclude_unset=True)
            result = await graph.ainvoke(
                Command(resume={active_interrupt.id: resume_value}), config=config
            )
            next_snapshot = await graph.aget_state(config)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("chat_resume_failed", thread_id=str(thread_id), error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail="Chat workflow is unavailable") from exc

    next_interrupt = _active_interrupt(next_snapshot)
    if next_interrupt is not None:
        next_values = next_snapshot.values
        return ChatResponse(
            thread_id=thread_id,
            reply="Your travel plan is ready for review.",
            status="interrupted",
            pending_approval={
                "interrupt_id": next_interrupt.id,
                "travel_plan": next_values.get("travel_plan"),
                "plan_revision_count": next_values.get("plan_revision_count", 0),
                "max_plan_revisions": MAX_PLAN_REVISIONS,
            },
        )
    if result.get("hitl_decision") == "reject":
        return ChatResponse(
            thread_id=thread_id,
            reply=result.get("final_response", "The travel plan was rejected."),
            status="rejected",
        )
    reply = result.get("final_response") or result["messages"][-1]["content"]
    return ChatResponse(thread_id=thread_id, reply=reply)

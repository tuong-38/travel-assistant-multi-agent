"""Opt-in asynchronous chat and approval-resume endpoints."""

from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
import structlog
logger = structlog.get_logger(__name__)
from pydantic import BaseModel

from app.api.chat import (
    ChatRequest,
    _active_interrupt,
    _has_checkpoint,
    _thread_resume_lock,
)
from app.core.config import get_settings
from app.graph.hitl import MAX_PLAN_REVISIONS, ResumeDecision

router = APIRouter(tags=["jobs"])


class QueuedJobResponse(BaseModel):
    job_id: UUID
    thread_id: UUID
    status: Literal["queued"] = "queued"


class JobStatusResponse(BaseModel):
    job_id: UUID
    thread_id: UUID
    status: Literal["queued", "running", "succeeded", "failed"]
    workflow_status: Literal["running", "waiting_for_input", "completed", "rejected"] | None = None
    reply: str | None = None
    pending_approval: dict[str, Any] | None = None
    error: str | None = None


def _config(thread_id: str, model_alias: str | None = None) -> dict[str, Any]:
    configurable: dict[str, str] = {"thread_id": thread_id}
    if model_alias:
        configurable["model_alias"] = model_alias
    return {"configurable": configurable}


@router.post("/chat/jobs", response_model=QueuedJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def enqueue_chat(payload: ChatRequest, request: Request) -> QueuedJobResponse:
    thread_id = payload.thread_id or uuid4()
    job_id = uuid4()
    model_alias = payload.model or get_settings().default_model
    graph = request.app.state.graph
    config = _config(str(thread_id), model_alias)
    try:
        async with _thread_resume_lock(request, str(thread_id)):
            if hasattr(graph, "aget_state"):
                snapshot = await graph.aget_state(config)
                if _active_interrupt(snapshot) is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="This thread is awaiting approval; use the resume endpoint",
                    )
            await request.app.state.jobs.enqueue(
                job_id=job_id,
                thread_id=thread_id,
                operation="chat",
                payload={"message": payload.message, "model_alias": model_alias},
            )
            logger.info("job_enqueued", job_id=job_id, thread_id=thread_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chat job queue is unavailable") from exc
    return QueuedJobResponse(job_id=job_id, thread_id=thread_id)


@router.post(
    "/chat/{thread_id}/resume/jobs",
    response_model=QueuedJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_resume_job(
    thread_id: UUID, payload: dict[str, Any], request: Request
) -> QueuedJobResponse:
    graph = request.app.state.graph
    config = _config(str(thread_id))
    try:
        decision = ResumeDecision.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid approval decision") from exc

    job_id = uuid4()
    try:
        async with _thread_resume_lock(request, str(thread_id)):
            snapshot = await graph.aget_state(config)
            if not _has_checkpoint(snapshot):
                raise HTTPException(status_code=404, detail="Conversation thread was not found")
            active_interrupt = _active_interrupt(snapshot)
            if active_interrupt is None:
                raise HTTPException(status_code=409, detail="Thread has no pending approval")
            if decision.interrupt_id != active_interrupt.id:
                raise HTTPException(status_code=409, detail="Approval interrupt is stale")
            values = snapshot.values
            if (
                decision.decision == "modify"
                and values.get("plan_revision_count", 0) >= MAX_PLAN_REVISIONS
            ):
                raise HTTPException(status_code=409, detail="Maximum plan revisions exceeded")
            await request.app.state.jobs.enqueue(
                job_id=job_id,
                thread_id=thread_id,
                operation="resume",
                payload={
                    "decision": decision.model_dump(exclude_unset=True),
                    "interrupt_id": active_interrupt.id,
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chat job queue is unavailable") from exc
    return QueuedJobResponse(job_id=job_id, thread_id=thread_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: UUID, request: Request) -> JobStatusResponse:
    try:
        job = await request.app.state.jobs.get(str(job_id))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Job status is unavailable") from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job was not found or has expired")

    response = JobStatusResponse(
        job_id=job_id,
        thread_id=UUID(job["thread_id"]),
        status=job["status"],
        error=job.get("error") or None,
    )
    graph = request.app.state.graph
    if hasattr(graph, "aget_state"):
        try:
            snapshot = await graph.aget_state(_config(job["thread_id"]))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Workflow state is unavailable") from exc
        if _has_checkpoint(snapshot):
            active_interrupt = _active_interrupt(snapshot)
            if active_interrupt is not None:
                values = snapshot.values
                response.workflow_status = "waiting_for_input"
                response.pending_approval = {
                    "interrupt_id": active_interrupt.id,
                    "travel_plan": values.get("travel_plan"),
                    "plan_revision_count": values.get("plan_revision_count", 0),
                    "max_plan_revisions": MAX_PLAN_REVISIONS,
                }
            elif values := getattr(snapshot, "values", None):
                if values.get("hitl_decision") == "reject":
                    response.workflow_status = "rejected"
                    response.reply = values.get("final_response")
                elif values.get("final_response"):
                    response.workflow_status = "completed"
                    response.reply = values["final_response"]
                elif response.status in {"queued", "running"}:
                    response.workflow_status = "running"
    return response

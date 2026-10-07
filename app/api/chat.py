from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(default=None, min_length=1)
    thread_id: UUID | None = None


class ChatResponse(BaseModel):
    thread_id: UUID
    reply: str


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    thread_id = payload.thread_id or uuid4()
    graph = request.app.state.graph
    model_alias = payload.model or get_settings().default_model
    try:
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": payload.message}],
                "user_request": payload.message,
            },
            config={
                "configurable": {
                    "thread_id": str(thread_id),
                    "model_alias": model_alias,
                }
            },
        )
    except Exception as exc:
        logger.error(
            "chat_request_failed",
            thread_id=str(thread_id),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Chat workflow is unavailable") from exc

    reply: str = result.get("final_response") or result["messages"][-1]["content"]
    return ChatResponse(thread_id=thread_id, reply=reply)

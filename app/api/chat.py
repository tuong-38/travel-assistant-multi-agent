from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: UUID | None = None


class ChatResponse(BaseModel):
    thread_id: UUID
    reply: str


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    thread_id = payload.thread_id or uuid4()
    graph = request.app.state.graph
    try:
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config={"configurable": {"thread_id": str(thread_id)}},
        )
    except Exception as exc:
        logger.error(
            "chat_request_failed",
            thread_id=str(thread_id),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Chat workflow is unavailable") from exc

    reply: str = result["messages"][-1]["content"]
    return ChatResponse(thread_id=thread_id, reply=reply)

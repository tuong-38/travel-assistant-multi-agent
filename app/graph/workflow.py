import time
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.graph.state import ChatState
from app.llm.factory import get_chat_model

logger = structlog.get_logger(__name__)


def build_graph(
    checkpointer: Any, chat_model: Any | None = None, model_alias: str | None = None
) -> Any:
    default_model_alias = model_alias or get_settings().default_model

    async def chat_node(
        state: ChatState, config: RunnableConfig
    ) -> dict[str, list[dict[str, str]]]:
        started_at = time.perf_counter()
        thread_id = config.get("configurable", {}).get("thread_id")
        selected_model_alias = (
            config.get("configurable", {}).get("model_alias") or default_model_alias
        )
        try:
            selected_chat_model = (
                chat_model if chat_model is not None else get_chat_model(selected_model_alias)
            )
            response = await selected_chat_model.ainvoke(state["messages"])
            if not isinstance(response.content, str):
                raise TypeError("Chat model returned non-text content")
        except Exception as exc:
            logger.error(
                "chat_model_invocation_failed",
                thread_id=thread_id,
                model_alias=selected_model_alias,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
                error_type=type(exc).__name__,
            )
            raise
        logger.info(
            "chat_model_invocation_completed",
            thread_id=thread_id,
            model_alias=selected_model_alias,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return {"messages": [{"role": "assistant", "content": response.content}]}

    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)
    return builder.compile(checkpointer=checkpointer)

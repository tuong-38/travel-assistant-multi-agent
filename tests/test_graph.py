import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.workflow import build_graph


class FakeChatModel:
    async def ainvoke(self, messages: list[dict[str, str]]) -> AIMessage:
        assert messages[-1]["content"] == "Hello"
        return AIMessage(content="A mocked response")


@pytest.mark.asyncio
async def test_graph_invokes_injected_chat_model() -> None:
    graph = build_graph(InMemorySaver(), chat_model=FakeChatModel(), model_alias="test_model")
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"configurable": {"thread_id": "test-thread"}},
    )
    assert result["messages"][-1] == {"role": "assistant", "content": "A mocked response"}

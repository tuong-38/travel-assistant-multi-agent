import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.workflow import build_graph


@pytest.mark.asyncio
async def test_graph_returns_deterministic_reply() -> None:
    graph = build_graph(InMemorySaver())
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"configurable": {"thread_id": "test-thread"}},
    )
    assert result["messages"][-1] == {"role": "assistant", "content": "You said: Hello"}

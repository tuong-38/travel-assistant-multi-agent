import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import workflow
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


@pytest.mark.asyncio
async def test_graph_selects_model_from_run_config(monkeypatch) -> None:
    selected_aliases: list[str] = []

    class AliasChatModel(FakeChatModel):
        async def ainvoke(self, messages: list[dict[str, str]]) -> AIMessage:
            return AIMessage(content="Response for selected model")

    def fake_get_chat_model(model_alias: str) -> AliasChatModel:
        selected_aliases.append(model_alias)
        return AliasChatModel()

    monkeypatch.setattr(workflow, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: type("Settings", (), {"default_model": "travel_general"})(),
    )
    graph = build_graph(InMemorySaver())

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"configurable": {"thread_id": "test-thread", "model_alias": "travel_local"}},
    )

    assert result["messages"][-1]["content"] == "Response for selected model"
    assert selected_aliases == ["travel_local"]

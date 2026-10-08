import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from structlog.testing import capture_logs

from app.graph.state import new_chat_execution_input
from app.graph.workflow import (
    DestinationExtractionError,
    PlannerOutputError,
    WorkflowRoutingError,
    build_graph,
)
from app.mcp.client import MCPToolClient
from mcp_servers.travel_tools.server import mcp

TRAVEL_PLAN_JSON = (
    '{"destination":"Hanoi","dates":"2026-11-01 to 2026-11-03",'
    '"duration":"3 days","budget":"moderate",'
    '"preferences":["food"],"constraints":["vegetarian"]}'
)


class RoutedFakeModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=next(self.outputs))


@pytest.mark.asyncio
async def test_graph_routes_to_planner_then_finishes() -> None:
    model = RoutedFakeModel(
        [
            '{"next":"planner"}',
            '{"destination":"Hanoi"}',
            "Hanoi lookup result.",
            '{"next":"planner"}',
            TRAVEL_PLAN_JSON,
            "Day 1: Hanoi.",
        ]
    )
    graph = build_graph(InMemorySaver(), chat_model=model, mcp_client=MCPToolClient(mcp))
    config = {"configurable": {"thread_id": "planner-thread", "model_alias": "travel_local"}}
    interrupted = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Plan three days in Hanoi"}],
            "user_request": "Plan three days in Hanoi",
            "model_alias": "travel_local",
        },
        config=config,
    )
    assert interrupted["travel_plan"]["destination"] == "Hanoi"
    snapshot = await graph.aget_state(config)
    interrupt_id = snapshot.interrupts[0].id
    result = await graph.ainvoke(
        Command(resume={interrupt_id: {"interrupt_id": interrupt_id, "decision": "approve"}}),
        config=config,
    )

    assert result["travel_plan"] == {
        "destination": "Hanoi",
        "dates": "2026-11-01 to 2026-11-03",
        "duration": "3 days",
        "budget": "moderate",
        "preferences": ["food"],
        "constraints": ["vegetarian"],
    }
    assert result["completed_agents"] == ["destination", "planner", "itinerary"]
    assert result["final_response"] == result["itinerary"]
    assert result["messages"][-1]["content"] == result["final_response"]


@pytest.mark.asyncio
async def test_graph_can_route_through_each_specialist() -> None:
    model = RoutedFakeModel(
        [
            '{"next":"destination"}',
            '{"destination":"Hanoi"}',
            "The destination is known for its history.",
            '{"next":"planner"}',
            TRAVEL_PLAN_JSON,
            "Day 1: Explore the old quarter.",
        ]
    )
    graph = build_graph(InMemorySaver(), chat_model=model, mcp_client=MCPToolClient(mcp))
    config = {"configurable": {"thread_id": "all-specialists"}}
    initial = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Plan a trip to Hanoi"}],
            "user_request": "Plan a trip to Hanoi",
        },
        config=config,
    )
    assert initial["destination_info"]["found"] is True
    snapshot = await graph.aget_state(config)
    interrupt_id = snapshot.interrupts[0].id
    result = await graph.ainvoke(
        Command(resume={interrupt_id: {"interrupt_id": interrupt_id, "decision": "approve"}}),
        config=config,
    )

    assert result["completed_agents"] == ["destination", "planner", "itinerary"]
    assert result["travel_plan"]["destination"] == "Hanoi"
    assert result["destination_info"]["found"] is True
    assert result["itinerary"] == "Day 1: Explore the old quarter."
    assert result["final_response"] == result["itinerary"]
    itinerary_input = model.calls[5][1].content
    assert json.loads(itinerary_input)["travel_plan"] == result["travel_plan"]


@pytest.mark.asyncio
async def test_destination_agent_extracts_only_destination_and_uses_fixed_mcp_tool() -> None:
    model = RoutedFakeModel(
        [
            '{"next":"destination"}',
            '{"destination":"Kyoto"}',
            "Kyoto lookup-backed answer.",
            '{"next":"FINISH","response":"Kyoto is ready."}',
        ]
    )
    graph = build_graph(InMemorySaver(), chat_model=model, mcp_client=MCPToolClient(mcp))
    result = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Tell me about Kyoto"}],
            "user_request": "Tell me about Kyoto",
        },
        config={"configurable": {"thread_id": "destination-mcp"}},
    )

    assert result["destination_info"]["destination"] == "Kyoto"
    assert result["destination_info"]["found"] is True
    assert "only the destination" in model.calls[1][0].content
    assert "search_destination" not in model.calls[1][0].content


@pytest.mark.asyncio
async def test_invalid_extracted_destination_is_rejected_before_mcp() -> None:
    class NeverCallMCP:
        async def search_destination(self, destination: str):
            raise AssertionError("MCP must not be called for invalid extraction")

    model = RoutedFakeModel(['{"next":"destination"}', '{"destination":"  "}'])
    graph = build_graph(InMemorySaver(), chat_model=model, mcp_client=NeverCallMCP())
    with pytest.raises(DestinationExtractionError):
        await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "Tell me about somewhere"}],
                "user_request": "Tell me about somewhere",
            },
            config={"configurable": {"thread_id": "invalid-destination"}},
        )
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_mcp_failure_stops_destination_agent_before_answer_generation() -> None:
    from app.mcp.client import MCPToolError

    class BrokenMCP:
        async def search_destination(self, destination: str):
            raise MCPToolError("MCP unavailable")

    model = RoutedFakeModel(['{"next":"destination"}', '{"destination":"Hanoi"}'])
    graph = build_graph(InMemorySaver(), chat_model=model, mcp_client=BrokenMCP())
    with pytest.raises(MCPToolError):
        await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "Tell me about Hanoi"}],
                "user_request": "Tell me about Hanoi",
            },
            config={"configurable": {"thread_id": "mcp-failure"}},
        )
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_invalid_planner_output_is_rejected() -> None:
    graph = build_graph(
        InMemorySaver(),
        chat_model=RoutedFakeModel(
            [
                '{"next":"planner"}',
                '{"destination":"Hanoi"}',
                "Destination",
                '{"next":"planner"}',
                "not a travel plan",
            ]
        ),
        mcp_client=MCPToolClient(mcp),
    )
    with pytest.raises(PlannerOutputError):
        await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "Plan a trip"}],
                "user_request": "Plan a trip",
            },
            config={"configurable": {"thread_id": "invalid-plan"}},
        )


@pytest.mark.asyncio
async def test_approval_node_rejects_invalid_direct_graph_resume() -> None:
    graph = build_graph(
        InMemorySaver(),
        chat_model=RoutedFakeModel(['{"next":"planner"}', TRAVEL_PLAN_JSON]),
    )
    config = {"configurable": {"thread_id": "invalid-resume"}}
    await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Plan a trip"}],
            "user_request": "Plan a trip",
            "destination_info": {
                "destination": "Hanoi",
                "found": True,
                "summary": "Known",
                "activities": [],
            },
            "completed_agents": ["destination"],
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)
    interrupt_id = snapshot.interrupts[0].id
    with pytest.raises(ValueError, match="Invalid approval resume value"):
        await graph.ainvoke(
            Command(resume={interrupt_id: {"interrupt_id": interrupt_id, "decision": "continue"}}),
            config=config,
        )


@pytest.mark.asyncio
async def test_supervisor_finishes_deterministically_after_itinerary() -> None:
    model = RoutedFakeModel(
        [
            '{"next":"destination"}',
            '{"destination":"Hanoi"}',
            "Hanoi lookup result.",
            '{"next":"planner"}',
            TRAVEL_PLAN_JSON,
            "Day 1: Explore the old quarter.",
        ]
    )
    graph = build_graph(
        InMemorySaver(),
        chat_model=model,
        mcp_client=MCPToolClient(mcp),
    )
    config = {"configurable": {"thread_id": "route-after-itinerary"}}
    interrupted = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Plan a trip to Hanoi"}],
            "user_request": "Plan a trip to Hanoi",
        },
        config=config,
    )
    assert interrupted["destination_info"]["destination"] == "Hanoi"
    assert interrupted["completed_agents"] == ["destination", "planner"]
    snapshot = await graph.aget_state(config)
    interrupt_id = snapshot.interrupts[0].id
    supervisor_calls_before_resume = sum(
        "Choose exactly one next step" in messages[0].content for messages in model.calls
    )
    assert supervisor_calls_before_resume == 2
    result = await graph.ainvoke(
        Command(resume={interrupt_id: {"interrupt_id": interrupt_id, "decision": "approve"}}),
        config=config,
    )
    final_snapshot = await graph.aget_state(config)
    assert result["current_agent"] == "FINISH"
    assert result["final_response"] == "Day 1: Explore the old quarter."
    assert result["completed_agents"] == ["destination", "planner", "itinerary"]
    supervisor_calls_after_resume = sum(
        "Choose exactly one next step" in messages[0].content for messages in model.calls
    )
    assert supervisor_calls_after_resume == supervisor_calls_before_resume
    assert len(model.calls) == 6
    assert final_snapshot.next == ()
    assert final_snapshot.interrupts == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_output",
    ["not json", '{"next":"arbitrary_node"}', '{"next":[]}', '{"next":"FINISH"}'],
)
async def test_invalid_supervisor_output_is_a_workflow_error(bad_output: str) -> None:
    graph = build_graph(InMemorySaver(), chat_model=RoutedFakeModel([bad_output]))
    with pytest.raises(WorkflowRoutingError):
        await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Hello"}], "user_request": "Hello"},
            config={"configurable": {"thread_id": "bad-route"}},
        )


@pytest.mark.asyncio
async def test_supervisor_logs_safe_decision_after_destination(monkeypatch) -> None:
    import app.graph.workflow as workflow

    sensitive_route = "private-user@example.com"
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "default_model": "travel_general",
                "travel_tools_mcp_url": "http://unused",
                "mcp_timeout_seconds": 5,
            },
        )(),
    )
    graph = build_graph(
        InMemorySaver(),
        chat_model=RoutedFakeModel([json.dumps({"next": sensitive_route})]),
    )
    with capture_logs() as logs:
        with pytest.raises(WorkflowRoutingError):
            await graph.ainvoke(
                {
                    "messages": [{"role": "user", "content": "Tell me about Hanoi"}],
                    "user_request": "Tell me about Hanoi",
                    "completed_agents": ["destination"],
                },
                config={"configurable": {"thread_id": "safe-log-thread"}},
            )

    event = next(log for log in logs if log["event"] == "supervisor_decision_after_destination")
    assert event["route_decision"] == "invalid"
    assert sensitive_route not in str(event)


@pytest.mark.asyncio
async def test_same_thread_appends_each_current_request_once() -> None:
    model = RoutedFakeModel(
        [
            '{"next":"FINISH","response":"First answer"}',
            '{"next":"FINISH","response":"Second answer"}',
        ]
    )
    graph = build_graph(InMemorySaver(), chat_model=model)
    config = {"configurable": {"thread_id": "continued-thread", "model_alias": "travel_general"}}
    for message in ("First request", "Second request"):
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}], "user_request": message},
            config=config,
        )

    user_messages = [item["content"] for item in result["messages"] if item["role"] == "user"]
    assert user_messages == ["First request", "Second request"]
    assert result["user_request"] == "Second request"
    assert config["configurable"]["thread_id"] == "continued-thread"


@pytest.mark.asyncio
async def test_new_execution_clears_old_workflow_state_and_preserves_history() -> None:
    model = RoutedFakeModel(
        [
            '{"next":"planner"}',
            TRAVEL_PLAN_JSON,
            "Day 1: Hanoi.",
            '{"next":"FINISH","response":"Follow-up received."}',
        ]
    )
    graph = build_graph(InMemorySaver(), chat_model=model)
    config = {"configurable": {"thread_id": "execution-boundary"}}
    first_request = "Plan a trip to Hanoi"
    interrupted = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": first_request}],
            "user_request": first_request,
            "destination_info": {
                "destination": "Hanoi",
                "found": True,
                "summary": "Known",
                "activities": [],
            },
            "completed_agents": ["destination"],
        },
        config=config,
    )
    assert interrupted["completed_agents"] == ["destination", "planner"]
    snapshot = await graph.aget_state(config)
    interrupt_id = snapshot.interrupts[0].id
    completed = await graph.ainvoke(
        Command(resume={interrupt_id: {"interrupt_id": interrupt_id, "decision": "approve"}}),
        config=config,
    )
    assert completed["completed_agents"] == ["destination", "planner", "itinerary"]

    follow_up = "Add a note about the first day"
    result = await graph.ainvoke(
        new_chat_execution_input(follow_up, "travel_local"),
        config=config,
    )

    user_messages = [
        message["content"] for message in result["messages"] if message["role"] == "user"
    ]
    assert user_messages == [first_request, follow_up]
    assert result["user_request"] == follow_up
    assert result["completed_agents"] == []
    assert result["destination_info"] is None
    assert result["travel_plan"] is None
    assert result["itinerary"] is None
    assert result["final_response"] == "Follow-up received."
    assert result["current_agent"] == "FINISH"
    assert len(model.calls) == 4


@pytest.mark.asyncio
async def test_selected_model_alias_is_used_for_each_graph_node(monkeypatch) -> None:
    import app.graph.workflow as workflow

    selected_aliases: list[str] = []
    supervisor_calls = 0

    class AliasModel:
        async def ainvoke(self, messages):
            nonlocal supervisor_calls
            if "Choose exactly one next step" in messages[0].content:
                supervisor_calls += 1
                if supervisor_calls == 1:
                    return AIMessage(content='{"next":"planner"}')
                return AIMessage(content='{"next":"FINISH","response":"Done"}')
            if "You are the travel planner" in messages[0].content:
                return AIMessage(content=TRAVEL_PLAN_JSON)
            return AIMessage(content="Day 1")

    def fake_get_chat_model(model_alias: str):
        selected_aliases.append(model_alias)
        return AliasModel()

    monkeypatch.setattr(workflow, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "default_model": "travel_general",
                "travel_tools_mcp_url": "http://unused",
                "mcp_timeout_seconds": 5,
            },
        )(),
    )
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "alias-thread", "model_alias": "travel_local"}}
    initial = await graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Plan a trip"}],
            "user_request": "Plan a trip",
            "destination_info": {
                "destination": "Hanoi",
                "found": True,
                "summary": "Known",
                "activities": [],
            },
            "completed_agents": ["destination"],
            "model_alias": "travel_local",
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)
    interrupt_id = snapshot.interrupts[0].id
    result = await graph.ainvoke(
        Command(resume={interrupt_id: {"interrupt_id": interrupt_id, "decision": "approve"}}),
        config=config,
    )
    assert result["final_response"] == result["itinerary"]
    assert initial["travel_plan"]["destination"] == "Hanoi"
    assert supervisor_calls == 1
    assert selected_aliases == ["travel_local"] * 3

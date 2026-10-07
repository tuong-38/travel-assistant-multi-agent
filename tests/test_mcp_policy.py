import pytest

from app.mcp.policy import TOOL_ALLOWLIST, ToolAuthorizationError, authorize_tool


def test_tool_allowlist_is_destination_only() -> None:
    assert TOOL_ALLOWLIST == {
        "destination": frozenset({"search_destination"}),
        "planner": frozenset(),
        "itinerary": frozenset(),
    }
    authorize_tool("destination", "search_destination")


@pytest.mark.parametrize("agent", ["planner", "itinerary", "supervisor", "unknown"])
def test_other_agents_cannot_call_mcp(agent: str) -> None:
    with pytest.raises(ToolAuthorizationError):
        authorize_tool(agent, "search_destination")


def test_unknown_tool_is_not_authorized_for_destination() -> None:
    with pytest.raises(ToolAuthorizationError):
        authorize_tool("destination", "arbitrary_tool")

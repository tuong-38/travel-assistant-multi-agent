SEARCH_DESTINATION = "search_destination"
TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "destination": frozenset({SEARCH_DESTINATION}),
    "planner": frozenset(),
    "itinerary": frozenset(),
}


class ToolAuthorizationError(PermissionError):
    """Raised when a graph agent requests a capability outside its policy."""


def authorize_tool(agent: str, tool_name: str) -> None:
    if tool_name not in TOOL_ALLOWLIST.get(agent, frozenset()):
        raise ToolAuthorizationError(f"Agent {agent!r} is not allowed to call {tool_name!r}")

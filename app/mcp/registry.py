from app.graph.state import DestinationInfo
from app.mcp.client import MCPToolClient
from app.mcp.policy import SEARCH_DESTINATION, authorize_tool


def validate_destination(destination: str) -> str:
    normalized = destination.strip()
    if not normalized or len(normalized) > 100 or any(ord(char) < 32 for char in normalized):
        raise ValueError("Extracted destination is invalid")
    return normalized


async def search_destination_for_agent(
    client: MCPToolClient, *, agent: str, destination: str
) -> DestinationInfo:
    authorize_tool(agent, SEARCH_DESTINATION)
    return await client.search_destination(validate_destination(destination))

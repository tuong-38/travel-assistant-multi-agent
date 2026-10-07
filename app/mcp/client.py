import asyncio
from typing import Any

from mcp import Client
from mcp.server.mcpserver import MCPServer

from app.graph.state import DestinationInfo
from app.mcp.policy import SEARCH_DESTINATION


class MCPToolError(RuntimeError):
    """Raised when the destination MCP capability cannot provide valid data."""


def _parse_destination_info(value: Any) -> DestinationInfo:
    expected = {"destination", "found", "summary", "activities"}
    if not isinstance(value, dict) or set(value) != expected:
        raise MCPToolError("MCP returned an invalid destination result shape")
    if not isinstance(value["destination"], str) or not isinstance(value["found"], bool):
        raise MCPToolError("MCP returned invalid destination result fields")
    if not isinstance(value["summary"], str) or not isinstance(value["activities"], list):
        raise MCPToolError("MCP returned invalid destination result fields")
    if not all(isinstance(activity, str) for activity in value["activities"]):
        raise MCPToolError("MCP returned invalid destination activities")
    if not value["found"] and (value["summary"] or value["activities"]):
        raise MCPToolError("MCP returned destination details for an unknown destination")
    return value


class MCPToolClient:
    def __init__(self, server: str | MCPServer, timeout_seconds: float = 5.0) -> None:
        self._server = server
        self._timeout_seconds = timeout_seconds

    async def search_destination(self, destination: str) -> DestinationInfo:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with Client(self._server) as client:
                    result = await client.call_tool(
                        SEARCH_DESTINATION, {"destination": destination}
                    )
        except TimeoutError as exc:
            raise MCPToolError("Destination MCP request timed out") from exc
        except Exception as exc:
            raise MCPToolError(
                "Destination MCP server is unavailable or returned a protocol error"
            ) from exc

        if result.is_error:
            raise MCPToolError("Destination MCP tool reported an error")
        if result.structured_content is None:
            raise MCPToolError("Destination MCP result did not include structured content")
        destination_info = _parse_destination_info(result.structured_content)
        if destination_info["destination"].casefold() != destination.strip().casefold():
            raise MCPToolError("MCP returned information for a different destination")
        return destination_info

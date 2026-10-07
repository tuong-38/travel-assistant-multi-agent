from mcp.server.mcpserver import MCPServer

from app.graph.state import DestinationInfo
from mcp_servers.travel_tools.tools import search_destination_data

mcp = MCPServer(name="travel-tools-mcp", version="0.1.0")


@mcp.tool()
def search_destination(destination: str) -> DestinationInfo:
    """Look up destination information in the deterministic local dataset."""
    return search_destination_data(destination)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)

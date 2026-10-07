import pytest
from mcp import Client

from mcp_servers.travel_tools.server import mcp
from mcp_servers.travel_tools.tools import search_destination_data


def test_static_destination_data_is_structured_and_deterministic() -> None:
    result = search_destination_data(" hanoi ")
    assert result == {
        "destination": "Hanoi",
        "found": True,
        "summary": "A historic Vietnamese city known for its Old Quarter and lakes.",
        "activities": ["Explore the Old Quarter", "Visit Hoan Kiem Lake"],
    }
    assert search_destination_data("Atlantis") == {
        "destination": "Atlantis",
        "found": False,
        "summary": "",
        "activities": [],
    }


@pytest.mark.asyncio
async def test_mcp_server_exposes_search_destination_as_structured_data() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("search_destination", {"destination": "Hanoi"})

    assert result.is_error is False
    assert result.structured_content == search_destination_data("Hanoi")

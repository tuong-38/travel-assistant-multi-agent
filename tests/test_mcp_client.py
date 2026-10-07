import asyncio
from types import SimpleNamespace

import pytest

from app.mcp.client import MCPToolClient, MCPToolError
from mcp_servers.travel_tools.server import mcp


@pytest.mark.asyncio
async def test_mcp_client_reads_valid_structured_result() -> None:
    result = await MCPToolClient(mcp).search_destination("Hanoi")
    assert result["destination"] == "Hanoi"
    assert result["found"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(
            is_error=True,
            structured_content={
                "destination": "Hanoi",
                "found": True,
                "summary": "invented",
                "activities": [],
            },
        ),
        SimpleNamespace(is_error=False, structured_content=None),
        SimpleNamespace(is_error=False, structured_content={"destination": "Hanoi"}),
    ],
)
async def test_mcp_client_rejects_errors_missing_or_invalid_content(monkeypatch, result) -> None:
    class FakeClient:
        def __init__(self, server) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def call_tool(self, name, arguments):
            assert name == "search_destination"
            assert arguments == {"destination": "Hanoi"}
            return result

    monkeypatch.setattr("app.mcp.client.Client", FakeClient)
    with pytest.raises(MCPToolError):
        await MCPToolClient("http://unused").search_destination("Hanoi")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connect", "protocol", "timeout"])
async def test_mcp_client_wraps_unavailable_and_timeout(monkeypatch, failure: str) -> None:
    class FakeClient:
        def __init__(self, server) -> None:
            pass

        async def __aenter__(self):
            if failure == "connect":
                raise OSError("server unavailable")
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def call_tool(self, name, arguments):
            if failure == "protocol":
                raise RuntimeError("protocol error")
            await asyncio.sleep(0.05)

    monkeypatch.setattr("app.mcp.client.Client", FakeClient)
    with pytest.raises(MCPToolError):
        await MCPToolClient("http://unused", timeout_seconds=0.01).search_destination("Hanoi")

"""Deterministic E2E over real Streamable HTTP MCP; PostgreSQL is not exercised."""

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import app.graph.workflow as workflow
from app.api.chat import router as chat_router
from app.mcp.client import MCPToolClient

TRAVEL_PLAN = {
    "destination": "Hanoi",
    "dates": "2026-11-01 to 2026-11-03",
    "duration": "3 days",
    "budget": "moderate",
    "preferences": ["food"],
    "constraints": ["vegetarian"],
}

DESTINATION_INFO = {
    "destination": "Hanoi",
    "found": True,
    "summary": "A historic Vietnamese city known for its Old Quarter and lakes.",
    "activities": ["Explore the Old Quarter", "Visit Hoan Kiem Lake"],
}


class DeterministicTravelModel:
    """A scripted model that records workflow inputs without contacting a provider."""

    def __init__(self) -> None:
        self.supervisor_calls = 0
        self.destination_info_seen: dict | None = None
        self.planner_input: dict | None = None
        self.planner_calls = 0
        self.itinerary_input: dict | None = None
        self.destination_answer = ""

    async def ainvoke(self, messages: list) -> AIMessage:
        system_prompt = messages[0].content
        if "Choose exactly one next step" in system_prompt:
            routes = ["destination", "planner", "FINISH"]
            route = routes[self.supervisor_calls]
            self.supervisor_calls += 1
            response = "The Hanoi plan is ready." if route == "FINISH" else None
            if self.supervisor_calls == 3:
                response = "Follow-up received."
            decision = {"next": route}
            if response is not None:
                decision["response"] = response
            return AIMessage(content=json.dumps(decision))

        if "Extract only the destination" in system_prompt:
            return AIMessage(content='{"destination":"Hanoi"}')

        request = json.loads(messages[1].content)
        if "using only facts in the structured lookup result" in system_prompt:
            self.destination_info_seen = request["destination_info"]
            self.destination_answer = (
                f"{request['destination_info']['destination']}: "
                f"{request['destination_info']['summary']}"
            )
            return AIMessage(content=self.destination_answer)

        if "You are the travel planner" in system_prompt:
            self.planner_input = request
            self.planner_calls += 1
            plan = dict(TRAVEL_PLAN)
            if self.planner_calls > 1:
                plan["budget"] = request["plan_revision_feedback"]["budget"]
            return AIMessage(content=json.dumps(plan))

        if "You are the itinerary specialist" in system_prompt:
            self.itinerary_input = request
            return AIMessage(content="Day 1: Explore the Old Quarter.")

        raise AssertionError(f"Unexpected model prompt: {system_prompt}")


class RecordingRealMCPClient:
    """Record arguments while delegating every call to the real MCP client."""

    def __init__(self, url: str) -> None:
        self.client = MCPToolClient(url)
        self.destinations: list[str] = []

    async def search_destination(self, destination: str) -> dict:
        self.destinations.append(destination)
        return await self.client.search_destination(destination)


class RecordingGraph:
    """Capture results/config while delegating to the compiled graph unchanged."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.results: list[dict] = []
        self.configs: list[dict] = []

    async def ainvoke(self, state: dict, config: dict) -> dict:
        result = await self.graph.ainvoke(state, config=config)
        self.results.append(result)
        self.configs.append(config)
        return result

    async def aget_state(self, config: dict):
        return await self.graph.aget_state(config)


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_mcp_server(port: int) -> tuple[subprocess.Popen[str], queue.Queue[str]]:
    server_code = (
        "import sys\n"
        "from mcp_servers.travel_tools.server import mcp\n"
        "mcp.run(transport='streamable-http', host='127.0.0.1', port=int(sys.argv[1]))\n"
    )
    process_env = os.environ.copy()
    process_env.pop("GEMINI_API_KEY", None)
    process_env.pop("LITELLM_API_KEY", None)
    process_env.pop("LITELLM_BASE_URL", None)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", server_code, str(port)],
        cwd=Path(__file__).resolve().parents[1],
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output.put(line.rstrip())

    threading.Thread(target=read_output, daemon=True).start()
    return process, output


def _wait_for_mcp_server(process: subprocess.Popen[str], output: queue.Queue[str]) -> None:
    deadline = time.monotonic() + 10
    captured: list[str] = []
    while time.monotonic() < deadline:
        if process.poll() is not None:
            while not output.empty():
                captured.append(output.get_nowait())
            raise AssertionError(f"MCP server exited during startup: {' | '.join(captured)}")
        try:
            line = output.get(timeout=max(0.01, deadline - time.monotonic()))
        except queue.Empty:
            break
        captured.append(line)
        if "Uvicorn running on http://127.0.0.1:" in line:
            return
    raise AssertionError(f"MCP server did not become ready: {' | '.join(captured)}")


def _stop_mcp_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if process.stdout is not None:
        process.stdout.close()


def test_phase4_fastapi_graph_real_mcp_e2e(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "get_settings",
        lambda: SimpleNamespace(
            default_model="travel_general",
            travel_tools_mcp_url="http://unused",
            mcp_timeout_seconds=5.0,
        ),
    )
    port = _available_local_port()
    process, output = _start_mcp_server(port)
    try:
        _wait_for_mcp_server(process, output)
        model = DeterministicTravelModel()
        mcp_client = RecordingRealMCPClient(f"http://127.0.0.1:{port}/mcp")
        compiled_graph = workflow.build_graph(
            InMemorySaver(), chat_model=model, mcp_client=mcp_client
        )
        recording_graph = RecordingGraph(compiled_graph)
        api = FastAPI()
        api.include_router(chat_router, prefix="/api/v1")
        api.state.graph = recording_graph
        api.state.testing = True
        thread_id = str(uuid4())

        with TestClient(api) as client:
            first_response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Plan a three-day trip to Hanoi",
                    "model": "travel_general",
                    "thread_id": thread_id,
                },
            )
            assert first_response.status_code == 200, first_response.text
            first_body = first_response.json()
            assert first_body["thread_id"] == thread_id
            assert first_body["reply"] == "Your travel plan is ready for review."
            assert first_body["status"] == "interrupted"
            initial_interrupt_id = first_body["pending_approval"]["interrupt_id"]

            modify_response = client.post(
                f"/api/v1/chat/{thread_id}/resume",
                json={
                    "interrupt_id": initial_interrupt_id,
                    "decision": "modify",
                    "changes": {"budget": "premium"},
                },
            )
            assert modify_response.status_code == 200, modify_response.text
            modified_body = modify_response.json()
            assert modified_body["status"] == "interrupted"
            assert modified_body["pending_approval"]["plan_revision_count"] == 1
            assert modified_body["pending_approval"]["travel_plan"]["budget"] == "premium"
            modified_interrupt_id = modified_body["pending_approval"]["interrupt_id"]

            approve_response = client.post(
                f"/api/v1/chat/{thread_id}/resume",
                json={"interrupt_id": modified_interrupt_id, "decision": "approve"},
            )
            assert approve_response.status_code == 200, approve_response.text
            assert approve_response.json() == {
                "thread_id": thread_id,
                "reply": "Day 1: Explore the Old Quarter.",
            }

            first_state = recording_graph.results[-1]
            assert first_state["current_agent"] == "FINISH"
            assert first_state["final_response"] == "Day 1: Explore the Old Quarter."
            assert first_state["destination_info"] == DESTINATION_INFO
            assert first_state["travel_plan"]["budget"] == "premium"
            assert first_state["itinerary"] == "Day 1: Explore the Old Quarter."
            assert first_state["completed_agents"] == ["destination", "planner", "itinerary"]
            assert mcp_client.destinations == ["Hanoi"]
            assert model.destination_info_seen == DESTINATION_INFO

            assert model.planner_input is not None
            assert any(
                item["role"] == "assistant" and item["content"] == model.destination_answer
                for item in model.planner_input["conversation"]
            )
            assert model.itinerary_input is not None
            assert model.itinerary_input["travel_plan"]["budget"] == "premium"
            assert model.planner_calls == 2

            second_response = client.post(
                "/api/v1/chat",
                json={
                    "message": "Add a note about the first day",
                    "model": "travel_general",
                    "thread_id": thread_id,
                },
            )
            assert second_response.status_code == 200, second_response.text
            assert second_response.json() == {
                "thread_id": thread_id,
                "reply": "Follow-up received.",
            }

        second_state = recording_graph.results[-1]
        user_messages = [
            message["content"] for message in second_state["messages"] if message["role"] == "user"
        ]
        assert user_messages == [
            "Plan a three-day trip to Hanoi",
            "Add a note about the first day",
        ]
        assert second_state["user_request"] == "Add a note about the first day"
        assert [config["configurable"]["thread_id"] for config in recording_graph.configs] == [
            thread_id,
            thread_id,
            thread_id,
            thread_id,
        ]
        assert mcp_client.destinations == ["Hanoi"]
    finally:
        _stop_mcp_server(process)

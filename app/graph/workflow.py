import json
import time
from typing import Any, Literal, cast

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.graph.state import TravelPlan, TravelState
from app.llm.factory import get_chat_model

logger = structlog.get_logger(__name__)
ALLOWED_ROUTES = {"destination", "planner", "itinerary", "FINISH"}
SPECIALIST_PROMPTS = {
    "destination": (
        "You are the destination specialist. Answer destination questions with useful, "
        "general characteristics, activities, and recommendations. Do not create a full itinerary."
    ),
    "planner": (
        "You are the travel planner. Organize the user's destination, dates, duration, "
        "budget, travel style, and constraints. Return only JSON with exactly these keys: "
        "destination, dates, duration, budget, preferences, constraints. Use strings or null "
        "for the first four values and arrays of strings for preferences and constraints."
    ),
    "itinerary": (
        "You are the itinerary specialist. Use the structured travel plan to create a "
        "practical day-by-day itinerary. Do not redo general requirement extraction."
    ),
}


class WorkflowRoutingError(ValueError):
    """Raised when the supervisor proposes an invalid or unsafe route."""


class PlannerOutputError(ValueError):
    """Raised when the planner does not return a valid structured travel plan."""


def _parse_travel_plan(content: str) -> TravelPlan:
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerOutputError("Planner returned malformed JSON") from exc
    expected_keys = {
        "destination",
        "dates",
        "duration",
        "budget",
        "preferences",
        "constraints",
    }
    if not isinstance(plan, dict) or set(plan) != expected_keys:
        raise PlannerOutputError("Planner output has an invalid travel plan shape")
    for key in ("destination", "dates", "duration", "budget"):
        if plan[key] is not None and not isinstance(plan[key], str):
            raise PlannerOutputError(f"Planner field {key} must be text or null")
    for key in ("preferences", "constraints"):
        if not isinstance(plan[key], list) or not all(isinstance(item, str) for item in plan[key]):
            raise PlannerOutputError(f"Planner field {key} must be a list of text values")
    return cast(TravelPlan, plan)


def _parse_supervisor_decision(content: Any) -> tuple[str, str | None]:
    if not isinstance(content, str):
        raise WorkflowRoutingError("Supervisor returned a non-text decision")
    try:
        decision = json.loads(content)
    except json.JSONDecodeError as exc:
        raise WorkflowRoutingError("Supervisor returned malformed JSON") from exc
    if not isinstance(decision, dict) or set(decision) - {"next", "response"}:
        raise WorkflowRoutingError("Supervisor decision has an invalid shape")
    route = decision.get("next")
    response = decision.get("response")
    if not isinstance(route, str) or route not in ALLOWED_ROUTES:
        raise WorkflowRoutingError("Supervisor proposed a route outside the allowlist")
    if response is not None and not isinstance(response, str):
        raise WorkflowRoutingError("Supervisor response must be text")
    if route == "FINISH" and not response:
        raise WorkflowRoutingError("Supervisor must provide a response when finishing")
    return cast(str, route), response


def build_graph(
    checkpointer: Any, chat_model: Any | None = None, model_alias: str | None = None
) -> Any:
    default_model_alias = model_alias or get_settings().default_model

    def model_for(config: RunnableConfig) -> Any:
        alias = config.get("configurable", {}).get("model_alias") or default_model_alias
        return chat_model if chat_model is not None else get_chat_model(alias)

    async def supervisor(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
        started_at = time.perf_counter()
        context = {
            "user_request": state.get("user_request", ""),
            "travel_plan": state.get("travel_plan"),
            "itinerary": state.get("itinerary"),
            "completed_agents": state.get("completed_agents", []),
            "previous_result": state.get("messages", [])[-1].get("content", ""),
        }
        prompt = (
            "Choose exactly one next step from destination, planner, itinerary, FINISH. "
            "Return only a JSON object with key next and, for FINISH, a concise response key. "
            "Do not repeat a completed specialist. An itinerary requires a travel plan. "
            f"Workflow context: {json.dumps(context, ensure_ascii=False)}"
        )
        response = await model_for(config).ainvoke(
            [SystemMessage(content=prompt), *state.get("messages", [])]
        )
        route, final_response = _parse_supervisor_decision(response.content)
        completed = state.get("completed_agents", [])
        if route in completed:
            raise WorkflowRoutingError("Supervisor attempted to repeat a completed specialist")
        if route == "itinerary" and not state.get("travel_plan"):
            raise WorkflowRoutingError("Supervisor selected itinerary before a travel plan exists")
        logger.info(
            "supervisor_routed",
            route=route,
            thread_id=config.get("configurable", {}).get("thread_id"),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        update: dict[str, Any] = {"current_agent": route}
        if route == "FINISH":
            update["final_response"] = final_response
            update["messages"] = [{"role": "assistant", "content": final_response}]
        return update

    async def specialist(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
        agent = state["current_agent"]
        if agent not in SPECIALIST_PROMPTS:
            raise WorkflowRoutingError("Invalid specialist node selection")
        response = await model_for(config).ainvoke(
            [
                SystemMessage(content=SPECIALIST_PROMPTS[agent]),
                HumanMessage(
                    content=json.dumps(
                        {
                            "user_request": state.get("user_request", ""),
                            "travel_plan": state.get("travel_plan"),
                            "itinerary": state.get("itinerary"),
                            "conversation": state.get("messages", []),
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        if not isinstance(response.content, str):
            raise TypeError("Agent model returned non-text content")
        travel_plan = _parse_travel_plan(response.content) if agent == "planner" else None
        completed = [*state.get("completed_agents", []), agent]
        update: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": response.content}],
            "completed_agents": completed,
        }
        if agent == "planner":
            update["travel_plan"] = travel_plan
        elif agent == "itinerary":
            update["itinerary"] = response.content
        return update

    def route_after_supervisor(
        state: TravelState,
    ) -> Literal["destination", "planner", "itinerary", "END"]:
        route = state["current_agent"]
        return "END" if route == "FINISH" else cast(Any, route)

    builder = StateGraph(TravelState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("destination", specialist)
    builder.add_node("planner", specialist)
    builder.add_node("itinerary", specialist)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"destination": "destination", "planner": "planner", "itinerary": "itinerary", "END": END},
    )
    for agent in SPECIALIST_PROMPTS:
        builder.add_edge(agent, "supervisor")
    return builder.compile(checkpointer=checkpointer)

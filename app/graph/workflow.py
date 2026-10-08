import json
import re
import time
from typing import Any, Literal, cast

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.core.config import get_settings
from app.graph.hitl import MAX_PLAN_REVISIONS, ResumeDecision, apply_plan_patch
from app.graph.state import DestinationInfo, TravelPlan, TravelState
from app.llm.factory import get_chat_model
from app.mcp.client import MCPToolClient
from app.mcp.registry import search_destination_for_agent, validate_destination

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
        "for the first four values and arrays of strings for preferences and constraints. "
        "When normalized plan_revision_feedback is present, incorporate those changes and "
        "preserve the other valid plan details."
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


class DestinationExtractionError(ValueError):
    """Raised when the destination-only extraction is malformed."""


def _is_planning_request(user_request: str) -> bool:
    """Recognize explicit itinerary/planning requests before enforcing destination lookup."""
    return bool(
        re.search(
            r"\b(plan|planning|itinerary|trip|travel|vacation|holiday)\b|"
            r"lịch trình|kế hoạch|du lịch|chuyến đi",
            user_request,
            flags=re.IGNORECASE,
        )
    )


def _parse_destination_extraction(content: Any) -> str | None:
    if not isinstance(content, str):
        raise DestinationExtractionError("Destination extraction was not text")
    try:
        extraction = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DestinationExtractionError("Destination extraction was malformed") from exc
    if not isinstance(extraction, dict) or set(extraction) != {"destination"}:
        raise DestinationExtractionError("Destination extraction must contain only destination")
    destination = extraction["destination"]
    if destination is None:
        return None
    if not isinstance(destination, str):
        raise DestinationExtractionError("Extracted destination must be text or null")
    try:
        return validate_destination(destination)
    except ValueError as exc:
        raise DestinationExtractionError("Extracted destination is invalid") from exc


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


def _safe_supervisor_decision(content: Any) -> str:
    """Return only an allowlisted route or a non-sensitive outcome category."""
    if not isinstance(content, str):
        return "non_text"
    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        return "malformed"
    if not isinstance(decision, dict):
        return "invalid"
    route = decision.get("next")
    return route if isinstance(route, str) and route in ALLOWED_ROUTES else "invalid"


def build_graph(
    checkpointer: Any,
    chat_model: Any | None = None,
    model_alias: str | None = None,
    mcp_client: MCPToolClient | None = None,
) -> Any:
    settings = get_settings()
    default_model_alias = model_alias or settings.default_model
    destination_client = mcp_client or MCPToolClient(
        settings.travel_tools_mcp_url, settings.mcp_timeout_seconds
    )

    def model_for(config: RunnableConfig) -> Any:
        alias = config.get("configurable", {}).get("model_alias") or default_model_alias
        return chat_model if chat_model is not None else get_chat_model(alias)

    async def supervisor(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
        completed = state.get("completed_agents", [])
        itinerary = state.get("itinerary")
        if "itinerary" in completed and isinstance(itinerary, str) and itinerary.strip():
            return {
                "current_agent": "FINISH",
                "final_response": itinerary,
                "messages": [{"role": "assistant", "content": itinerary}],
            }

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
        if "destination" in completed:
            logger.info(
                "supervisor_decision_after_destination",
                route_decision=_safe_supervisor_decision(response.content),
                thread_id=config.get("configurable", {}).get("thread_id"),
            )
        route, final_response = _parse_supervisor_decision(response.content)
        if route in completed:
            raise WorkflowRoutingError("Supervisor attempted to repeat a completed specialist")
        if route == "itinerary" and not state.get("travel_plan"):
            raise WorkflowRoutingError("Supervisor selected itinerary before a travel plan exists")
        if "itinerary" in completed and route != "FINISH":
            raise WorkflowRoutingError("Supervisor must finish after itinerary generation")
        if route == "itinerary" and state.get("hitl_decision") != "approve":
            raise WorkflowRoutingError("Supervisor selected itinerary before plan approval")
        if (
            route == "planner"
            and _is_planning_request(state.get("user_request", ""))
            and not state.get("destination_info")
        ):
            if "destination" in completed:
                raise WorkflowRoutingError(
                    "Planning requires destination information, but destination lookup is complete"
                )
            route = "destination"
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

    async def destination_agent(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
        travel_plan = state.get("travel_plan")
        destination = travel_plan.get("destination") if travel_plan else None
        if destination is not None:
            try:
                destination = validate_destination(destination)
            except (AttributeError, ValueError) as exc:
                raise DestinationExtractionError("Travel plan destination is invalid") from exc
        else:
            extraction = await model_for(config).ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Extract only the destination named in the user request. "
                            'Return exactly JSON like {"destination":"Hanoi"}; if none is named, '
                            'return {"destination":null}. Do not name or call tools.'
                        )
                    ),
                    HumanMessage(content=state.get("user_request", "")),
                ]
            )
            destination = _parse_destination_extraction(extraction.content)

        if destination is None:
            answer = "Which destination would you like information about?"
            return {
                "messages": [{"role": "assistant", "content": answer}],
                "completed_agents": [*state.get("completed_agents", []), "destination"],
            }

        destination_info: DestinationInfo = await search_destination_for_agent(
            destination_client, agent="destination", destination=destination
        )
        if not destination_info["found"]:
            answer = f"I don't have destination information for {destination_info['destination']}."
        else:
            response = await model_for(config).ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Answer the user's destination question using only facts in the "
                            "structured lookup result. Do not add unsupported facts."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "user_request": state.get("user_request", ""),
                                "destination_info": destination_info,
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            if not isinstance(response.content, str):
                raise TypeError("Destination model returned non-text content")
            answer = response.content
        return {
            "destination_info": destination_info,
            "messages": [{"role": "assistant", "content": answer}],
            "completed_agents": [*state.get("completed_agents", []), "destination"],
        }

    async def specialist(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
        agent = state["current_agent"]
        if agent not in {"planner", "itinerary"}:
            raise WorkflowRoutingError("Invalid specialist node selection")
        planner_context = {
            "user_request": state.get("user_request", ""),
            "travel_plan": state.get("travel_plan"),
            "plan_revision_feedback": state.get("plan_revision_feedback"),
            "itinerary": state.get("itinerary"),
            "conversation": state.get("messages", []),
        }
        response = await model_for(config).ainvoke(
            [
                SystemMessage(content=SPECIALIST_PROMPTS[agent]),
                HumanMessage(content=json.dumps(planner_context, ensure_ascii=False)),
            ]
        )
        if not isinstance(response.content, str):
            raise TypeError("Agent model returned non-text content")
        travel_plan = _parse_travel_plan(response.content) if agent == "planner" else None
        completed = state.get("completed_agents", [])
        if agent not in completed:
            completed = [*completed, agent]
        update: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": response.content}],
            "completed_agents": completed,
        }
        if agent == "planner":
            update["travel_plan"] = travel_plan
            update["plan_revision_feedback"] = None
        elif agent == "itinerary":
            update["itinerary"] = response.content
        return update

    async def approval(state: TravelState) -> dict[str, Any]:
        decision_value = interrupt(
            {
                "type": "travel_plan_approval",
                "approval_type": "travel_plan",
                "travel_plan": state.get("travel_plan"),
                "plan_revision_count": state.get("plan_revision_count", 0),
                "max_plan_revisions": MAX_PLAN_REVISIONS,
            }
        )
        try:
            decision = ResumeDecision.model_validate(decision_value)
        except Exception as exc:
            raise ValueError("Invalid approval resume value") from exc
        if decision.decision == "modify":
            revision_count = state.get("plan_revision_count", 0)
            if revision_count >= MAX_PLAN_REVISIONS:
                raise ValueError("Maximum plan revisions exceeded")
            plan = state.get("travel_plan")
            if plan is None or decision.changes is None:
                raise ValueError("Cannot modify a missing travel plan")
            updated_plan = apply_plan_patch(plan, decision.changes)
            return {
                "travel_plan": updated_plan,
                "plan_revision_count": revision_count + 1,
                "plan_revision_feedback": decision.changes.model_dump(exclude_unset=True),
                "hitl_decision": "modify",
                "current_agent": "planner",
            }
        if decision.decision == "reject":
            reply = "The travel plan was rejected."
            return {
                "hitl_decision": "reject",
                "current_agent": "REJECTED",
                "final_response": reply,
                "messages": [{"role": "assistant", "content": reply}],
            }
        return {"hitl_decision": "approve", "current_agent": "itinerary"}

    def route_after_supervisor(
        state: TravelState,
    ) -> Literal["destination", "planner", "itinerary", "END"]:
        route = state["current_agent"]
        return "END" if route == "FINISH" else cast(Any, route)

    builder = StateGraph(TravelState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("destination", destination_agent)
    builder.add_node("planner", specialist)
    builder.add_node("itinerary", specialist)
    builder.add_node("approval", approval)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"destination": "destination", "planner": "planner", "itinerary": "itinerary", "END": END},
    )
    builder.add_edge("destination", "supervisor")
    builder.add_edge("planner", "approval")
    builder.add_conditional_edges(
        "approval",
        lambda state: (
            "planner"
            if state["hitl_decision"] == "modify"
            else END
            if state["hitl_decision"] == "reject"
            else "itinerary"
        ),
        {"planner": "planner", "itinerary": "itinerary", END: END},
    )
    builder.add_edge("itinerary", "supervisor")
    return builder.compile(checkpointer=checkpointer)

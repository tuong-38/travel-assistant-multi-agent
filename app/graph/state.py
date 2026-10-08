import operator
from typing import Annotated, TypedDict


class TravelPlan(TypedDict):
    destination: str | None
    dates: str | None
    duration: str | None
    budget: str | None
    preferences: list[str]
    constraints: list[str]


class DestinationInfo(TypedDict):
    destination: str
    found: bool
    summary: str
    activities: list[str]


class TravelState(TypedDict, total=False):
    """Conversation history and state scoped to the current workflow execution."""

    messages: Annotated[list[dict[str, str]], operator.add]
    user_request: str
    travel_plan: TravelPlan | None
    destination_info: DestinationInfo | None
    itinerary: str | None
    current_agent: str | None
    completed_agents: list[str]
    final_response: str | None
    hitl_decision: str | None
    plan_revision_count: int
    plan_revision_feedback: dict[str, object] | None
    model_alias: str
    execution_job_id: str | None


def new_chat_execution_input(
    message: str,
    model_alias: str,
    execution_job_id: str | None = None,
) -> TravelState:
    """Initialize one new execution while the message reducer retains thread history."""
    return {
        "messages": [{"role": "user", "content": message}],
        "user_request": message,
        "model_alias": model_alias,
        "execution_job_id": execution_job_id,
        "completed_agents": [],
        "destination_info": None,
        "travel_plan": None,
        "itinerary": None,
        "current_agent": None,
        "final_response": None,
        "hitl_decision": None,
        "plan_revision_count": 0,
        "plan_revision_feedback": None,
    }

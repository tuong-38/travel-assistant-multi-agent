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
    """Conversation context and compact structured travel workflow state."""

    messages: Annotated[list[dict[str, str]], operator.add]
    user_request: str
    travel_plan: TravelPlan
    destination_info: DestinationInfo
    itinerary: str
    current_agent: str
    completed_agents: list[str]
    final_response: str

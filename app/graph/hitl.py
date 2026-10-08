from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.graph.state import TravelPlan

MAX_PLAN_REVISIONS = 3


class TravelPlanPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: str | None = None
    dates: str | None = None
    duration: str | None = None
    budget: str | None = None
    preferences: list[str] | None = None
    constraints: list[str] | None = None


class ResumeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interrupt_id: str = Field(min_length=1)
    decision: Literal["approve", "modify", "reject"]
    changes: TravelPlanPatch | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> "ResumeDecision":
        if self.decision == "modify":
            if self.changes is None or not self.changes.model_fields_set:
                raise ValueError("modify requires a non-empty changes object")
            if any(
                field in self.changes.model_fields_set and getattr(self.changes, field) is None
                for field in ("preferences", "constraints")
            ):
                raise ValueError("preferences and constraints must be lists")
        elif "changes" in self.model_fields_set:
            raise ValueError("changes are only allowed with modify")
        return self


def apply_plan_patch(plan: TravelPlan, patch: TravelPlanPatch) -> TravelPlan:
    updated = {**plan, **patch.model_dump(exclude_unset=True)}
    return updated  # type: ignore[return-value]

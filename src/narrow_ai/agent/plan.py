"""Plan contract: a short, ordered list of ordinary intent decisions."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, model_validator

from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import IntentDecision, IntentName

MAX_STEPS = 5

# Steps that make sense on their own but never inside a multi-step plan: an
# "unsupported" or "clarify" step means the plan as a whole isn't understood.
NOT_PLANNABLE = {IntentName.UNSUPPORTED, IntentName.CLARIFY_REQUEST}


class PlanRejected(ValueError):
    """The whole plan is refused; no step of it runs."""


class PlanStep(BaseModel):
    model_config = {"extra": "forbid"}

    decision: IntentDecision
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = {"extra": "forbid"}

    goal: str = Field(..., min_length=1, max_length=2000)
    steps: list[PlanStep] = Field(..., min_length=1, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def _deps_point_backwards(self) -> Plan:
        for i, step in enumerate(self.steps):
            if any(not (0 <= d < i) for d in step.depends_on):
                raise ValueError(f"step {i} depends on a step that is not earlier")
        return self


def validate_plan(plan: Plan, registry: IntentRegistry, *, max_steps: int = MAX_STEPS) -> Plan:
    """All-or-nothing check of every step before anything runs (fail closed)."""
    if len(plan.steps) > max_steps:
        raise PlanRejected(f"plan has {len(plan.steps)} steps (max {max_steps})")
    for i, step in enumerate(plan.steps):
        d = step.decision
        try:
            spec = registry.get(d.intent)
        except KeyError as exc:
            raise PlanRejected(f"step {i}: unknown intent {d.intent}") from exc
        if not spec.implemented:
            raise PlanRejected(f"step {i}: intent not implemented: {d.intent.value}")
        if d.intent in NOT_PLANNABLE:
            raise PlanRejected(f"step {i}: {d.intent.value} can't be part of a plan")
        try:
            d.validated_params()
        except ValidationError as exc:
            raise PlanRejected(f"step {i}: invalid parameters for {d.intent.value}") from exc
    return plan

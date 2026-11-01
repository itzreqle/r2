"""User confirmation gate for privileged actions."""

from __future__ import annotations

from dataclasses import dataclass

from narrow_ai.intents.schemas import IntentDecision


@dataclass
class ConfirmationRequired:
    decision: IntentDecision
    prompt: str

    @classmethod
    def for_decision(cls, decision: IntentDecision) -> ConfirmationRequired:
        return cls(
            decision=decision,
            prompt=(
                f"Confirm action '{decision.intent}' "
                f"with parameters {decision.parameters}? [y/N]"
            ),
        )

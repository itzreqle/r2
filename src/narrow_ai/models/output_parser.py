"""Parse model text into validated IntentDecision."""

from __future__ import annotations

from narrow_ai.intents.classifier import IntentParseError, parse_intent_decision
from narrow_ai.intents.schemas import IntentDecision


def parse_model_output(text: str) -> IntentDecision:
    return parse_intent_decision(text)


__all__ = ["parse_model_output", "IntentParseError"]

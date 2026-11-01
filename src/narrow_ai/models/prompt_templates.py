"""Prompt helpers kept separate from weights."""

from __future__ import annotations

from narrow_ai.intents.registry import IntentRegistry


def build_system_prompt(base: str, registry: IntentRegistry) -> str:
    implemented = ", ".join(s.name.value for s in registry.implemented())
    return (
        f"{base.strip()}\n\n"
        f"Currently implemented intents: {implemented}.\n"
        "Output ONLY valid JSON with keys intent, parameters, requires_confirmation."
    )

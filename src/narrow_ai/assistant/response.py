"""User-facing response formatting."""

from __future__ import annotations

from dataclasses import dataclass

from narrow_ai.intents.schemas import IntentDecision
from narrow_ai.tools.executor import ToolResult


@dataclass
class AssistantResponse:
    text: str
    decision: IntentDecision | None = None
    tool_result: ToolResult | None = None
    state: str = "idle"
    needs_confirmation: bool = False

    def spoken_text(self) -> str:
        # Keep TTS brief
        return self.text[:800]

"""User-facing response formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from narrow_ai.intents.schemas import IntentDecision
from narrow_ai.tools.result import ToolResult


@dataclass
class AssistantResponse:
    text: str
    decision: IntentDecision | None = None
    tool_result: ToolResult | None = None
    state: str = "idle"
    needs_confirmation: bool = False
    speech: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    # Single-use id that must be echoed back to approve (HTTP API); "" = none pending.
    confirmation_id: str = ""
    expires_in: float = 0.0
    # Set when the request became a multi-step run (Run.to_dict(), never the id).
    run: dict[str, Any] | None = None

    def spoken_text(self) -> str:
        """Short Siri-style sentence for TTS."""
        return (self.speech or self.text)[:300]

    def to_dict(self) -> dict[str, Any]:
        if self.run is not None:
            intent: str | None = "multi_step"
            ok = self.run.get("status") in {"completed", "awaiting_confirmation"}
        else:
            intent = self.decision.intent.value if self.decision else None
            ok = bool(self.tool_result and self.tool_result.ok)
        out = {
            "text": self.text,
            "speech": self.spoken_text(),
            "intent": intent,
            "parameters": self.decision.parameters if self.decision else {},
            "ok": ok,
            "state": str(self.state),
            "needs_confirmation": self.needs_confirmation,
            "metrics": self.metrics,
            "confirmation_id": self.confirmation_id or None,
            "expires_in": round(self.expires_in, 1) if self.confirmation_id else None,
        }
        if self.run is not None:
            out["run"] = self.run
        return out

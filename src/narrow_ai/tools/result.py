"""Result type shared by every tool handler."""

from __future__ import annotations

from dataclasses import dataclass

from narrow_ai.intents.schemas import IntentName


@dataclass
class ToolResult:
    ok: bool
    intent: IntentName
    message: str
    data: dict | None = None
    needs_confirmation: bool = False
    speech: str = ""  # short sentence for TTS (Siri-style); falls back to message

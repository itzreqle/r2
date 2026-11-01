"""Conversation state helpers."""

from __future__ import annotations

from enum import StrEnum


class AssistantState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    ERROR = "error"
    AWAITING_CONFIRMATION = "awaiting_confirmation"

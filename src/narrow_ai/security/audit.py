"""In-memory audit trail for tool attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class AuditEvent:
    timestamp: str
    intent: str
    allowed: bool
    detail: str


@dataclass
class AuditLog:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, intent: str, allowed: bool, detail: str) -> None:
        self.events.append(
            AuditEvent(
                timestamp=datetime.now(UTC).isoformat(),
                intent=intent,
                allowed=allowed,
                detail=detail[:500],
            )
        )

"""User confirmation gate for privileged actions.

A privileged decision is parked as a :class:`PendingAction` with a random,
single-use, expiring id. Over the HTTP API the client must echo that id to
approve, so a confirmation is bound to the exact prompt the user was shown:
it cannot be replayed, approve a different (newer) action, or be issued by a
caller that never saw the prompt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from narrow_ai.intents.schemas import IntentDecision

Clock = Callable[[], float]


def decision_digest(decision: IntentDecision) -> str:
    payload = json.dumps(decision.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class ConfirmCheck(StrEnum):
    OK = "ok"
    MISSING = "missing"  # no id supplied
    MISMATCH = "mismatch"  # wrong id (stale UI, other caller)
    EXPIRED = "expired"
    TAMPERED = "tampered"  # decision changed since it was parked


@dataclass
class PendingAction:
    decision: IntentDecision
    ttl_sec: float = 120.0
    clock: Clock = field(default=time.monotonic, repr=False)
    id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    digest: str = ""
    created: float = 0.0

    def __post_init__(self) -> None:
        self.created = self.clock()
        self.digest = decision_digest(self.decision)

    @property
    def expires_in(self) -> float:
        return max(0.0, self.created + self.ttl_sec - self.clock())

    @property
    def expired(self) -> bool:
        return self.clock() >= self.created + self.ttl_sec

    def check(self, confirmation_id: str | None, *, require_id: bool = True) -> ConfirmCheck:
        if self.expired:
            return ConfirmCheck.EXPIRED
        if require_id:
            if not confirmation_id:
                return ConfirmCheck.MISSING
            if not hmac.compare_digest(confirmation_id.encode(), self.id.encode()):
                return ConfirmCheck.MISMATCH
        if decision_digest(self.decision) != self.digest:
            return ConfirmCheck.TAMPERED
        return ConfirmCheck.OK


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

from __future__ import annotations

from narrow_ai.intents.registry import default_registry
from narrow_ai.intents.schemas import IntentDecision, IntentName
from narrow_ai.settings import Settings
from narrow_ai.tools.permissions import PermissionGate


def test_remember_requires_confirmation():
    gate = PermissionGate(Settings(), default_registry())
    d = IntentDecision(
        intent=IntentName.REMEMBER_PREFERENCE,
        parameters={"key": "a", "value": "b"},
    )
    out = gate.check(d)
    assert out.allowed
    assert out.needs_confirmation


def test_unimplemented_blocked():
    gate = PermissionGate(Settings(), default_registry())
    d = IntentDecision(intent=IntentName.RUN_TESTS, parameters={"command_id": "x"})
    out = gate.check(d)
    assert not out.allowed

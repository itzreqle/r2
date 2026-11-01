from __future__ import annotations

from pathlib import Path

import pytest

from narrow_ai.intents.registry import default_registry
from narrow_ai.intents.schemas import IntentDecision, IntentName, PermissionLevel
from narrow_ai.settings import Settings
from narrow_ai.tools.executor import ToolExecutor
from narrow_ai.tools.handlers import default_tools
from narrow_ai.tools.registry import ToolRegistry, ToolSpec
from narrow_ai.tools.result import ToolResult


def test_handlers_exist_for_exactly_the_implemented_intents():
    implemented = {s.name for s in default_registry().implemented()}
    assert default_tools().intents() == implemented
    assert IntentName.RUN_TESTS not in default_tools().intents()


def test_duplicate_registration_is_an_error():
    spec = ToolSpec(IntentName.GIT_STATUS, lambda p, c: ToolResult(True, IntentName.GIT_STATUS, ""))
    with pytest.raises(ValueError):
        ToolRegistry([spec, spec])


def _settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.workspace.path = str(tmp_path)
    s.tools.allow_open_app = True
    s.tools.approved_apps = ["Cursor"]
    return s


def test_unapproved_app_is_refused_before_asking_for_confirmation(tmp_path):
    """Regression: R2 used to ask "Open Calculator? Say yes or no." and refuse after yes."""
    res = ToolExecutor(_settings(tmp_path), default_registry()).execute(
        IntentDecision(intent=IntentName.OPEN_APP, parameters={"app_name": "Calculator"})
    )
    assert not res.ok and not res.needs_confirmation
    assert "approved app list" in res.message


def test_approved_app_still_asks_for_confirmation(tmp_path):
    res = ToolExecutor(_settings(tmp_path), default_registry()).execute(
        IntentDecision(intent=IntentName.OPEN_APP, parameters={"app_name": "cursor"})
    )
    assert res.needs_confirmation and not res.ok


def test_missing_preference_store_is_refused_before_confirmation(tmp_path):
    res = ToolExecutor(_settings(tmp_path), default_registry(), preferences=None).execute(
        IntentDecision(
            intent=IntentName.REMEMBER_PREFERENCE, parameters={"key": "k", "value": "v"}
        )
    )
    assert not res.ok and not res.needs_confirmation


def test_intent_without_handler_fails_closed(tmp_path):
    """Even if an intent were marked implemented, no handler means no execution."""
    registry = default_registry()
    spec = registry.get(IntentName.RUN_TESTS)
    registry._by_name[IntentName.RUN_TESTS] = spec.model_copy(
        update={"implemented": True, "permission": PermissionLevel.READ}
    )
    s = _settings(tmp_path)
    s.tools.allow_run_tests = True
    res = ToolExecutor(s, registry).execute(
        IntentDecision(intent=IntentName.RUN_TESTS, parameters={"command_id": "unit"})
    )
    assert not res.ok and "No executor" in res.message

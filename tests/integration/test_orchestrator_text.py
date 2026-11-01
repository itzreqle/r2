from __future__ import annotations

from pathlib import Path

from narrow_ai.assistant.orchestrator import AssistantOrchestrator
from narrow_ai.intents.schemas import IntentName
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.models.fake_backend import FakeModelBackend
from narrow_ai.settings import Settings


def test_help_and_search(tmp_path: Path):
    (tmp_path / "demo.py").write_text("print(1)\n", encoding="utf-8")
    settings = Settings()
    settings.workspace.path = str(tmp_path)
    settings.memory.db_path = str(tmp_path / "prefs.db")
    orch = AssistantOrchestrator(
        settings,
        FakeModelBackend(),
        preferences=PreferenceStore(tmp_path / "prefs.db"),
    )
    help_resp = orch.handle_text("What can you do?")
    assert help_resp.decision and help_resp.decision.intent == IntentName.GENERAL_HELP
    assert "git_status" in help_resp.text

    search = orch.handle_text("Find Python files in the project.")
    assert search.decision and search.decision.intent == IntentName.SEARCH_FILES
    assert "demo.py" in search.text


def test_remember_needs_confirm(tmp_path: Path):
    settings = Settings()
    settings.workspace.path = str(tmp_path)
    prefs = PreferenceStore(tmp_path / "prefs.db")
    orch = AssistantOrchestrator(settings, FakeModelBackend(), preferences=prefs)
    first = orch.handle_text("Remember preferred_editor=Cursor")
    assert first.needs_confirmation
    assert prefs.get("preferred_editor") is None
    second = orch.handle_text("yes")
    assert prefs.get("preferred_editor") == "Cursor"
    assert second.tool_result and second.tool_result.ok

from __future__ import annotations

from pathlib import Path

from narrow_ai.assistant.orchestrator import AssistantOrchestrator
from narrow_ai.intents.schemas import IntentDecision, IntentName
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.models.fake_backend import FakeModelBackend
from narrow_ai.security.confirmation import ConfirmCheck, PendingAction
from narrow_ai.settings import Settings


def _decision(app: str = "Cursor") -> IntentDecision:
    return IntentDecision(intent=IntentName.OPEN_APP, parameters={"app_name": app})


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_pending_action_checks():
    clock = Clock()
    pa = PendingAction(_decision(), ttl_sec=10, clock=clock)
    assert len(pa.id) >= 20
    assert pa.check(pa.id) == ConfirmCheck.OK
    assert pa.check(None) == ConfirmCheck.MISSING
    assert pa.check("") == ConfirmCheck.MISSING
    assert pa.check(pa.id[:-1] + "x") == ConfirmCheck.MISMATCH
    assert pa.check(None, require_id=False) == ConfirmCheck.OK
    clock.t += 10
    assert pa.expired and pa.check(pa.id) == ConfirmCheck.EXPIRED
    assert pa.check(None, require_id=False) == ConfirmCheck.EXPIRED


def test_pending_ids_are_unique():
    ids = {PendingAction(_decision()).id for _ in range(200)}
    assert len(ids) == 200


def test_tampered_decision_is_rejected():
    pa = PendingAction(_decision())
    pa.decision.parameters["app_name"] = "Terminal"  # mutated after the prompt was shown
    assert pa.check(pa.id) == ConfirmCheck.TAMPERED


def _orch(tmp_path: Path, clock: Clock) -> tuple[AssistantOrchestrator, PreferenceStore]:
    settings = Settings()
    settings.workspace.path = str(tmp_path)
    settings.security.confirmation_ttl_sec = 30
    prefs = PreferenceStore(tmp_path / "prefs.db")
    return (
        AssistantOrchestrator(settings, FakeModelBackend(), preferences=prefs, clock=clock),
        prefs,
    )


def test_orchestrator_confirm_is_single_use(tmp_path: Path):
    orch, prefs = _orch(tmp_path, Clock())
    first = orch.handle_text("remember editor=Cursor")
    assert first.confirmation_id and orch.pending_action.id == first.confirmation_id
    assert orch.confirm(first.confirmation_id).tool_result.ok
    assert prefs.get("editor") == "Cursor"
    assert orch.confirm(first.confirmation_id).text == "Nothing to confirm."


def test_orchestrator_expiry_in_process(tmp_path: Path):
    clock = Clock()
    orch, prefs = _orch(tmp_path, clock)
    orch.handle_text("remember editor=Cursor")
    clock.t += 31
    out = orch.handle_text("yes")  # trusted in-process UI, but still expired
    assert "expired" in out.text and prefs.get("editor") is None
    assert orch.pending is None


def test_new_request_supersedes_pending(tmp_path: Path):
    orch, prefs = _orch(tmp_path, Clock())
    first = orch.handle_text("remember editor=Cursor")
    orch.handle_text("git status")
    assert orch.pending is None
    assert orch.confirm(first.confirmation_id).text == "Nothing to confirm."
    assert prefs.get("editor") is None


def test_confirmations_are_audited(tmp_path: Path):
    orch, _ = _orch(tmp_path, Clock())
    first = orch.handle_text("remember editor=Cursor")
    orch.confirm("wrong")
    orch.handle_text("remember editor=Vim")
    orch.handle_text("no")
    details = [e.detail for e in orch.audit.events]
    assert "confirmation mismatch" in details and "cancelled" in details
    assert first.confirmation_id not in " ".join(details)

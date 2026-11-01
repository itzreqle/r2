"""Permission decisions live in application code, not in the LLM."""

from __future__ import annotations

from dataclasses import dataclass

from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import IntentDecision, IntentName, PermissionLevel
from narrow_ai.settings import Settings


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str
    needs_confirmation: bool = False


class PermissionGate:
    def __init__(self, settings: Settings, registry: IntentRegistry) -> None:
        self.settings = settings
        self.registry = registry

    def check(self, decision: IntentDecision) -> PermissionDecision:
        try:
            spec = self.registry.get(decision.intent)
        except KeyError:
            return PermissionDecision(False, "Unknown intent")

        if not spec.implemented:
            return PermissionDecision(False, f"Intent not implemented: {decision.intent}")

        tools = self.settings.tools
        if decision.intent == IntentName.GIT_STATUS and not tools.allow_git_status:
            return PermissionDecision(False, "git_status disabled in config")
        if decision.intent == IntentName.GIT_DIFF_SUMMARY and not tools.allow_git_diff:
            return PermissionDecision(False, "git_diff_summary disabled in config")
        if decision.intent == IntentName.SEARCH_FILES and not tools.allow_search_files:
            return PermissionDecision(False, "search_files disabled in config")
        if decision.intent == IntentName.READ_FILE and not tools.allow_read_file:
            return PermissionDecision(False, "read_file disabled in config")
        if decision.intent == IntentName.RUN_TESTS and not tools.allow_run_tests:
            return PermissionDecision(False, "run_tests disabled in config")
        if decision.intent == IntentName.OPEN_APP and not tools.allow_open_app:
            return PermissionDecision(False, "open_app disabled in config")

        needs_confirm = decision.requires_confirmation or (
            spec.permission in {PermissionLevel.WRITE, PermissionLevel.EXECUTE}
            and self.settings.security.require_confirmation_for_writes
        )
        if decision.intent in {IntentName.REMEMBER_PREFERENCE, IntentName.RUN_TESTS}:
            needs_confirm = True  # run_tests executes the project's own code

        return PermissionDecision(True, "ok", needs_confirmation=needs_confirm)

"""Validate + execute allowlisted tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import (
    IntentDecision,
    IntentName,
    RecallPreferenceParams,
    RememberPreferenceParams,
    SearchFilesParams,
)
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.settings import Settings
from narrow_ai.tools.file_tools import search_files
from narrow_ai.tools.git_tools import GitToolError, git_status
from narrow_ai.tools.permissions import PermissionGate

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool
    intent: IntentName
    message: str
    data: dict | None = None
    needs_confirmation: bool = False


class ToolExecutor:
    def __init__(
        self,
        settings: Settings,
        registry: IntentRegistry,
        preferences: PreferenceStore | None = None,
        *,
        confirmed: bool = False,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.gate = PermissionGate(settings, registry)
        self.preferences = preferences
        self.confirmed = confirmed

    def execute(self, decision: IntentDecision) -> ToolResult:
        perm = self.gate.check(decision)
        if not perm.allowed:
            return ToolResult(False, decision.intent, perm.reason)

        if perm.needs_confirmation and not self.confirmed:
            return ToolResult(
                False,
                decision.intent,
                f"Confirmation required before: {decision.intent}",
                needs_confirmation=True,
                data={"decision": decision.model_dump()},
            )

        workspace = self.settings.workspace_path()
        timeout = self.settings.tools.command_timeout_sec
        max_chars = self.settings.tools.max_output_chars

        try:
            if decision.intent == IntentName.GENERAL_HELP:
                return ToolResult(True, decision.intent, self.registry.help_text())

            if decision.intent == IntentName.GIT_STATUS:
                text = git_status(workspace, timeout=timeout, max_chars=max_chars)
                return ToolResult(True, decision.intent, text)

            if decision.intent == IntentName.SEARCH_FILES:
                params = SearchFilesParams.model_validate(decision.parameters)
                hits = search_files(workspace, params.pattern)
                msg = (
                    f"Found {len(hits)} match(es) for {params.pattern}:\n"
                    + "\n".join(hits)
                    if hits
                    else f"No files matching {params.pattern}"
                )
                return ToolResult(True, decision.intent, msg, data={"files": hits})

            if decision.intent == IntentName.REMEMBER_PREFERENCE:
                if self.preferences is None:
                    return ToolResult(False, decision.intent, "Preference store unavailable")
                params = RememberPreferenceParams.model_validate(decision.parameters)
                self.preferences.set(params.key, params.value)
                return ToolResult(
                    True,
                    decision.intent,
                    f"Remembered preference '{params.key}'.",
                )

            if decision.intent == IntentName.RECALL_PREFERENCE:
                if self.preferences is None:
                    return ToolResult(False, decision.intent, "Preference store unavailable")
                params = RecallPreferenceParams.model_validate(decision.parameters)
                value = self.preferences.get(params.key)
                if value is None:
                    return ToolResult(False, decision.intent, f"No preference named '{params.key}'")
                return ToolResult(True, decision.intent, f"{params.key} = {value}")

            if decision.intent == IntentName.CLARIFY_REQUEST:
                q = decision.parameters.get("question", "Could you clarify?")
                return ToolResult(True, decision.intent, str(q))

            if decision.intent == IntentName.UNSUPPORTED:
                reason = decision.parameters.get("reason", "Unsupported request")
                return ToolResult(True, decision.intent, str(reason))

            return ToolResult(
                False,
                decision.intent,
                f"No executor registered for {decision.intent}",
            )
        except GitToolError as exc:
            return ToolResult(False, decision.intent, str(exc))
        except Exception as exc:  # noqa: BLE001 — surface tool failures cleanly
            logger.exception("tool execution failed")
            return ToolResult(False, decision.intent, f"Tool error: {exc}")

"""Validate + execute allowlisted tools. Model output is never authorization.

The one execution path, shared by single requests and multi-step runs::

    PermissionGate.check → parameter validation → tool precheck
        → confirmation (if required and not yet given) → tool handler
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import ValidationError

from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import IntentDecision, IntentName
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.settings import Settings
from narrow_ai.tools.application_tools import AppToolError
from narrow_ai.tools.git_tools import GitToolError
from narrow_ai.tools.handlers import default_tools, normalize_key
from narrow_ai.tools.path_guard import PathEscapeError
from narrow_ai.tools.permissions import PermissionGate
from narrow_ai.tools.registry import ToolContext, ToolRegistry
from narrow_ai.tools.result import ToolResult

logger = logging.getLogger(__name__)

__all__ = ["ToolExecutor", "ToolResult", "normalize_key"]

_DEFAULT_TOOLS: ToolRegistry | None = None


def _default_tools() -> ToolRegistry:
    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        _DEFAULT_TOOLS = default_tools()
    return _DEFAULT_TOOLS


class ToolExecutor:
    def __init__(
        self,
        settings: Settings,
        registry: IntentRegistry,
        preferences: PreferenceStore | None = None,
        *,
        confirmed: bool = False,
        answerer: Callable[[str], str] | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.gate = PermissionGate(settings, registry)
        self.preferences = preferences
        self.confirmed = confirmed
        # Text-only Q&A via the local model. Output is displayed/spoken, never executed.
        self.answerer = answerer
        self.tools = tools or _default_tools()

    def describe(self, decision: IntentDecision) -> str:
        """Human confirmation prompt for a privileged action."""
        p = decision.parameters
        if decision.intent == IntentName.OPEN_APP:
            return f"Open {p.get('app_name')}?"
        if decision.intent == IntentName.REMEMBER_PREFERENCE:
            return f"Remember {normalize_key(str(p.get('key', '')))} = {p.get('value')}?"
        return f"Run {decision.intent.value} with {p}?"

    def _context(self) -> ToolContext:
        return ToolContext(
            settings=self.settings,
            intents=self.registry,
            workspace=self.settings.workspace_path(),
            timeout=self.settings.tools.command_timeout_sec,
            max_chars=self.settings.tools.max_output_chars,
            preferences=self.preferences,
            answerer=self.answerer,
        )

    def execute(self, decision: IntentDecision) -> ToolResult:
        perm = self.gate.check(decision)
        if not perm.allowed:
            return ToolResult(False, decision.intent, perm.reason, speech="I can't do that here.")

        try:
            params = decision.validated_params()
        except ValidationError as exc:
            return ToolResult(
                False,
                decision.intent,
                f"Invalid parameters: {exc.errors()[0].get('msg', exc)}",
                speech="Sorry, I didn't get the details of that request.",
            )

        intent = decision.intent
        spec = self.tools.get(intent)
        if spec is None:  # fail closed: a schema without a handler can't run
            return ToolResult(False, intent, f"No executor registered for {intent}")

        try:
            ctx = self._context()
            if spec.precheck is not None:
                refusal = spec.precheck(params, ctx)
                if refusal is not None:  # never ask to confirm something we'd refuse
                    return refusal

            if perm.needs_confirmation and not self.confirmed:
                prompt = self.describe(decision)
                return ToolResult(
                    False,
                    intent,
                    f"{prompt} Say yes to confirm or no to cancel.",
                    needs_confirmation=True,
                    data={"decision": decision.model_dump()},
                    speech=f"{prompt} Say yes or no.",
                )

            return spec.handler(params, ctx)
        except (GitToolError, PathEscapeError, AppToolError, FileNotFoundError) as exc:
            msg = str(exc)
            speech = (
                "That folder isn't a Git repository."
                if "not a git repository" in msg
                else ("I couldn't do that.")
            )
            if isinstance(exc, FileNotFoundError):
                msg, speech = f"File not found: {exc}", "I couldn't find that file."
            return ToolResult(False, intent, msg, speech=speech)
        except Exception as exc:  # noqa: BLE001 — surface tool failures cleanly
            logger.exception("tool execution failed")
            return ToolResult(False, intent, f"Tool error: {exc}", speech="Something went wrong.")

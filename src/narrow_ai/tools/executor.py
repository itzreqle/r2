"""Validate + execute allowlisted tools. Model output is never authorization."""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import (
    AnswerQuestionParams,
    IntentDecision,
    IntentName,
    OpenAppParams,
    ReadFileParams,
    RecallPreferenceParams,
    RememberPreferenceParams,
    SearchFilesParams,
)
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.settings import Settings
from narrow_ai.tools.application_tools import AppToolError, open_approved_app
from narrow_ai.tools.file_tools import read_workspace_file, search_files
from narrow_ai.tools.git_tools import GitToolError, git_diff_stat, git_status
from narrow_ai.tools.path_guard import PathEscapeError
from narrow_ai.tools.permissions import PermissionGate

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool
    intent: IntentName
    message: str
    data: dict | None = None
    needs_confirmation: bool = False
    speech: str = ""  # short sentence for TTS (Siri-style); falls back to message


def normalize_key(key: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return key[:64] or "default"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _git_status_speech(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(("branch:", "##"))]
    branch = text.splitlines()[0].removeprefix("branch: ").strip() if text else "unknown"
    if not lines:
        return f"You're on {branch} and the working tree is clean."
    return f"You're on {branch} with {_plural(len(lines), 'changed file')}."


class ToolExecutor:
    def __init__(
        self,
        settings: Settings,
        registry: IntentRegistry,
        preferences: PreferenceStore | None = None,
        *,
        confirmed: bool = False,
        answerer: Callable[[str], str] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.gate = PermissionGate(settings, registry)
        self.preferences = preferences
        self.confirmed = confirmed
        # Text-only Q&A via the local model. Output is displayed/spoken, never executed.
        self.answerer = answerer

    def describe(self, decision: IntentDecision) -> str:
        """Human confirmation prompt for a privileged action."""
        p = decision.parameters
        if decision.intent == IntentName.OPEN_APP:
            return f"Open {p.get('app_name')}?"
        if decision.intent == IntentName.REMEMBER_PREFERENCE:
            return f"Remember {normalize_key(str(p.get('key', '')))} = {p.get('value')}?"
        return f"Run {decision.intent.value} with {p}?"

    def execute(self, decision: IntentDecision) -> ToolResult:
        perm = self.gate.check(decision)
        if not perm.allowed:
            return ToolResult(False, decision.intent, perm.reason, speech="I can't do that here.")

        try:
            decision.validated_params()
        except ValidationError as exc:
            return ToolResult(
                False,
                decision.intent,
                f"Invalid parameters: {exc.errors()[0].get('msg', exc)}",
                speech="Sorry, I didn't get the details of that request.",
            )

        if perm.needs_confirmation and not self.confirmed:
            prompt = self.describe(decision)
            return ToolResult(
                False,
                decision.intent,
                f"{prompt} Say yes to confirm or no to cancel.",
                needs_confirmation=True,
                data={"decision": decision.model_dump()},
                speech=f"{prompt} Say yes or no.",
            )

        workspace = self.settings.workspace_path()
        timeout = self.settings.tools.command_timeout_sec
        max_chars = self.settings.tools.max_output_chars
        intent = decision.intent

        try:
            if intent == IntentName.GENERAL_HELP:
                return ToolResult(
                    True,
                    intent,
                    self.registry.help_text(),
                    speech=(
                        "I can check your git status and diff, find and read files, open apps, "
                        "remember preferences, and answer developer questions. Try: "
                        "what's my git status, find the Swift files, or how do I undo a commit."
                    ),
                )

            if intent == IntentName.GIT_STATUS:
                text = git_status(workspace, timeout=timeout, max_chars=max_chars)
                return ToolResult(True, intent, text, speech=_git_status_speech(text))

            if intent == IntentName.GIT_DIFF_SUMMARY:
                text = git_diff_stat(workspace, timeout=timeout, max_chars=max_chars)
                summary = [ln for ln in text.splitlines() if "changed" in ln]
                speech = summary[-1].strip() if summary else "There are no changes."
                return ToolResult(True, intent, text, speech=speech)

            if intent == IntentName.SEARCH_FILES:
                params = SearchFilesParams.model_validate(decision.parameters)
                hits = search_files(workspace, params.pattern)
                if not hits:
                    msg = f"No files matching {params.pattern}"
                    return ToolResult(True, intent, msg, data={"files": []}, speech=msg + ".")
                msg = f"Found {len(hits)} match(es) for {params.pattern}:\n" + "\n".join(hits)
                more = " or more" if len(hits) >= 50 else ""
                return ToolResult(
                    True,
                    intent,
                    msg,
                    data={"files": hits},
                    speech=f"I found {_plural(len(hits), 'file')}{more} matching {params.pattern}.",
                )

            if intent == IntentName.READ_FILE:
                params = ReadFileParams.model_validate(decision.parameters)
                text = read_workspace_file(workspace, params.path, max_chars=min(max_chars, 4000))
                n_lines = text.count("\n") + 1
                return ToolResult(
                    True,
                    intent,
                    f"--- {params.path} ---\n{text}",
                    data={"path": params.path},
                    speech=f"Here's {params.path}, {_plural(n_lines, 'line')} shown.",
                )

            if intent == IntentName.OPEN_APP:
                params = OpenAppParams.model_validate(decision.parameters)
                approved = self.settings.tools.approved_apps
                match = {a.lower(): a for a in approved}.get(params.app_name.strip().lower())
                if match is None:
                    return ToolResult(
                        False,
                        intent,
                        f"{params.app_name} is not on the approved app list: {', '.join(approved)}",
                        speech=f"{params.app_name} isn't on my approved app list.",
                    )
                msg = open_approved_app(match, approved)
                return ToolResult(True, intent, msg, speech=msg + ".")

            if intent == IntentName.REMEMBER_PREFERENCE:
                if self.preferences is None:
                    return ToolResult(False, intent, "Preference store unavailable")
                params = RememberPreferenceParams.model_validate(decision.parameters)
                key = normalize_key(params.key)
                self.preferences.set(key, params.value)
                return ToolResult(
                    True,
                    intent,
                    f"Remembered {key} = {params.value}.",
                    speech=f"Okay, I'll remember your {key.replace('_', ' ')}.",
                )

            if intent == IntentName.RECALL_PREFERENCE:
                if self.preferences is None:
                    return ToolResult(False, intent, "Preference store unavailable")
                params = RecallPreferenceParams.model_validate(decision.parameters)
                key = normalize_key(params.key)
                value = self.preferences.get(key)
                if value is None:
                    close = difflib.get_close_matches(
                        key, self.preferences.list_keys(), n=1, cutoff=0.5
                    )
                    if close:
                        key, value = close[0], self.preferences.get(close[0])
                if value is None:
                    return ToolResult(
                        False,
                        intent,
                        f"No preference named '{key}'",
                        speech="I don't have that saved yet.",
                    )
                return ToolResult(
                    True,
                    intent,
                    f"{key} = {value}",
                    speech=f"Your {key.replace('_', ' ')} is {value}.",
                )

            if intent == IntentName.ANSWER_QUESTION:
                params = AnswerQuestionParams.model_validate(decision.parameters)
                if self.answerer is None:
                    return ToolResult(False, intent, "Question answering is unavailable.")
                answer = self.answerer(params.question).strip()
                if not answer:
                    return ToolResult(
                        False,
                        intent,
                        "I don't have an answer for that.",
                        speech="I don't have an answer for that.",
                    )
                return ToolResult(True, intent, answer, speech=answer)

            if intent == IntentName.CLARIFY_REQUEST:
                q = str(decision.parameters.get("question", "Could you clarify?"))
                return ToolResult(True, intent, q, speech=q)

            if intent == IntentName.UNSUPPORTED:
                reason = str(decision.parameters.get("reason", "Unsupported request"))
                return ToolResult(
                    True,
                    intent,
                    f"Sorry, I can't help with that. ({reason})",
                    speech="Sorry, that's outside what I can do.",
                )

            return ToolResult(False, intent, f"No executor registered for {intent}")
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

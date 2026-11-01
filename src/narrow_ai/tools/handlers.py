"""One handler per implemented intent (moved verbatim from ToolExecutor's if-chain).

Handlers receive *validated* parameters and run only after the executor has
passed the permission gate, any precheck, and confirmation.
"""

from __future__ import annotations

import difflib
import re

from pydantic import BaseModel

from narrow_ai.intents.schemas import (
    AnswerQuestionParams,
    ClarifyRequestParams,
    IntentName,
    OpenAppParams,
    ReadFileParams,
    RecallPreferenceParams,
    RememberPreferenceParams,
    SearchFilesParams,
    UnsupportedParams,
)
from narrow_ai.tools.application_tools import open_approved_app
from narrow_ai.tools.file_tools import display_name, read_workspace_file, search_files
from narrow_ai.tools.git_tools import git_diff_stat, git_status
from narrow_ai.tools.registry import ToolContext, ToolRegistry, ToolSpec
from narrow_ai.tools.result import ToolResult

I = IntentName  # noqa: E741 - short alias for the table below


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


# --------------------------------------------------------------------- read-only
def general_help(_: BaseModel, ctx: ToolContext) -> ToolResult:
    return ToolResult(
        True,
        I.GENERAL_HELP,
        ctx.intents.help_text(),
        speech=(
            "I can check your git status and diff, find and read files, open apps, "
            "remember preferences, and answer developer questions. Try: "
            "what's my git status, find the Swift files, or how do I undo a commit."
        ),
    )


def git_status_tool(_: BaseModel, ctx: ToolContext) -> ToolResult:
    text = git_status(ctx.workspace, timeout=ctx.timeout, max_chars=ctx.max_chars)
    return ToolResult(True, I.GIT_STATUS, text, speech=_git_status_speech(text))


def git_diff_tool(_: BaseModel, ctx: ToolContext) -> ToolResult:
    text = git_diff_stat(ctx.workspace, timeout=ctx.timeout, max_chars=ctx.max_chars)
    summary = [ln for ln in text.splitlines() if "changed" in ln]
    speech = summary[-1].strip() if summary else "There are no changes."
    return ToolResult(True, I.GIT_DIFF_SUMMARY, text, speech=speech)


def search_files_tool(params: SearchFilesParams, ctx: ToolContext) -> ToolResult:
    hits = search_files(ctx.workspace, params.pattern)
    if not hits:
        msg = f"No files matching {params.pattern}"
        return ToolResult(True, I.SEARCH_FILES, msg, data={"files": []}, speech=msg + ".")
    listing = "\n".join(display_name(h) for h in hits)
    msg = f"Found {len(hits)} match(es) for {params.pattern}:\n{listing}"
    more = " or more" if len(hits) >= 50 else ""
    return ToolResult(
        True,
        I.SEARCH_FILES,
        msg,
        data={"files": hits},
        speech=f"I found {_plural(len(hits), 'file')}{more} matching {params.pattern}.",
    )


def read_file_tool(params: ReadFileParams, ctx: ToolContext) -> ToolResult:
    text = read_workspace_file(ctx.workspace, params.path, max_chars=min(ctx.max_chars, 4000))
    n_lines = text.count("\n") + 1
    return ToolResult(
        True,
        I.READ_FILE,
        f"--- {params.path} ---\n{text}",
        data={"path": params.path},
        speech=f"Here's {params.path}, {_plural(n_lines, 'line')} shown.",
    )


# --------------------------------------------------------------------- privileged
def _approved_app(params: OpenAppParams, ctx: ToolContext) -> str | None:
    approved = ctx.settings.tools.approved_apps
    return {a.lower(): a for a in approved}.get(params.app_name.strip().lower())


def open_app_precheck(params: OpenAppParams, ctx: ToolContext) -> ToolResult | None:
    if _approved_app(params, ctx) is not None:
        return None
    approved = ctx.settings.tools.approved_apps
    return ToolResult(
        False,
        I.OPEN_APP,
        f"{params.app_name} is not on the approved app list: {', '.join(approved)}",
        speech=f"{params.app_name} isn't on my approved app list.",
    )


def open_app_tool(params: OpenAppParams, ctx: ToolContext) -> ToolResult:
    refusal = open_app_precheck(params, ctx)  # re-checked at execution time too
    if refusal is not None:
        return refusal
    match = _approved_app(params, ctx)
    assert match is not None
    msg = open_approved_app(match, ctx.settings.tools.approved_apps)
    return ToolResult(True, I.OPEN_APP, msg, speech=msg + ".")


def needs_preferences(intent: IntentName):
    def precheck(_: BaseModel, ctx: ToolContext) -> ToolResult | None:
        if ctx.preferences is None:
            return ToolResult(False, intent, "Preference store unavailable")
        return None

    return precheck


def remember_tool(params: RememberPreferenceParams, ctx: ToolContext) -> ToolResult:
    if ctx.preferences is None:
        return ToolResult(False, I.REMEMBER_PREFERENCE, "Preference store unavailable")
    key = normalize_key(params.key)
    ctx.preferences.set(key, params.value)
    return ToolResult(
        True,
        I.REMEMBER_PREFERENCE,
        f"Remembered {key} = {params.value}.",
        speech=f"Okay, I'll remember your {key.replace('_', ' ')}.",
    )


def recall_tool(params: RecallPreferenceParams, ctx: ToolContext) -> ToolResult:
    if ctx.preferences is None:
        return ToolResult(False, I.RECALL_PREFERENCE, "Preference store unavailable")
    key = normalize_key(params.key)
    value = ctx.preferences.get(key)
    if value is None:
        close = difflib.get_close_matches(key, ctx.preferences.list_keys(), n=1, cutoff=0.5)
        if close:
            key, value = close[0], ctx.preferences.get(close[0])
    if value is None:
        return ToolResult(
            False,
            I.RECALL_PREFERENCE,
            f"No preference named '{key}'",
            speech="I don't have that saved yet.",
        )
    return ToolResult(
        True,
        I.RECALL_PREFERENCE,
        f"{key} = {value}",
        speech=f"Your {key.replace('_', ' ')} is {value}.",
    )


# --------------------------------------------------------------------- talk only
def answer_tool(params: AnswerQuestionParams, ctx: ToolContext) -> ToolResult:
    if ctx.answerer is None:
        return ToolResult(False, I.ANSWER_QUESTION, "Question answering is unavailable.")
    answer = ctx.answerer(params.question).strip()  # displayed/spoken, never executed
    if not answer:
        return ToolResult(
            False,
            I.ANSWER_QUESTION,
            "I don't have an answer for that.",
            speech="I don't have an answer for that.",
        )
    return ToolResult(True, I.ANSWER_QUESTION, answer, speech=answer)


def clarify_tool(params: ClarifyRequestParams, _: ToolContext) -> ToolResult:
    return ToolResult(True, I.CLARIFY_REQUEST, params.question, speech=params.question)


def unsupported_tool(params: UnsupportedParams, _: ToolContext) -> ToolResult:
    return ToolResult(
        True,
        I.UNSUPPORTED,
        f"Sorry, I can't help with that. ({params.reason})",
        speech="Sorry, that's outside what I can do.",
    )


def default_tools() -> ToolRegistry:
    """Handlers for every implemented intent. ``run_tests`` / ``explain_test_output``
    have none, so even a misconfigured registry could not execute them."""
    return ToolRegistry(
        [
            ToolSpec(I.GENERAL_HELP, general_help),
            ToolSpec(I.GIT_STATUS, git_status_tool),
            ToolSpec(I.GIT_DIFF_SUMMARY, git_diff_tool),
            ToolSpec(I.SEARCH_FILES, search_files_tool),
            ToolSpec(I.READ_FILE, read_file_tool),
            ToolSpec(I.OPEN_APP, open_app_tool, precheck=open_app_precheck),
            ToolSpec(
                I.REMEMBER_PREFERENCE,
                remember_tool,
                precheck=needs_preferences(I.REMEMBER_PREFERENCE),
            ),
            ToolSpec(I.RECALL_PREFERENCE, recall_tool),
            ToolSpec(I.ANSWER_QUESTION, answer_tool),
            ToolSpec(I.CLARIFY_REQUEST, clarify_tool),
            ToolSpec(I.UNSUPPORTED, unsupported_tool),
        ]
    )

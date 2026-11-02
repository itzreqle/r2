"""Bridge between single requests (/ask, voice) and multi-step runs.

Clients stay thin: they send text (or audio) with ``allow_runs`` and get back an
ordinary response that may carry a ``run``. The confirmation id of a paused run is
handed out exactly like a single-step confirmation, so the Confirm button, a typed
"yes" and a spoken "yes" all work unchanged.
"""

from __future__ import annotations

from typing import Any

from narrow_ai.agent.plan import Plan
from narrow_ai.agent.planner import split_clauses
from narrow_ai.agent.runner import Run, RunStatus, StepStatus
from narrow_ai.assistant.response import AssistantResponse

_QUESTION_START = (
    "how ", "what ", "what's ", "why ", "when ", "where ", "which ", "who ", "is ", "are ",
    "does ", "do ", "can ", "could ", "should ", "explain ", "tell me about ",
)


def looks_multistep(text: str) -> bool:
    """Cheap pre-check before planning: explicit connectors, or "and"/a comma in
    something that doesn't read like a question. A miss only means the normal
    single-step path handles the request."""
    t = " ".join(text.lower().split())
    if len(split_clauses(t)) > 1:
        return True
    return (" and " in t or "," in t) and not t.startswith(_QUESTION_START)


class FixedPlanner:
    """Hands an already-built plan (or an already-made rejection) to
    ``AgentRunner.start``, so planning happens exactly once."""

    def __init__(self, plan: Plan | None = None, *, rejection: Exception | None = None) -> None:
        self._plan = plan
        self._rejection = rejection

    def plan(self, text: str) -> Plan:
        if self._rejection is not None:
            raise self._rejection
        assert self._plan is not None
        return self._plan


_STEP_MARK = {
    StepStatus.OK: "✓", StepStatus.FAILED: "✗", StepStatus.AWAITING_CONFIRMATION: "?",
    StepStatus.SKIPPED: "–", StepStatus.CANCELLED: "⊘", StepStatus.RUNNING: "…",
    StepStatus.PENDING: "·",
}


def run_response(run: Run, *, since: int = 0) -> AssistantResponse:
    """Turn a run into a normal response. ``since``: first step index that ran during
    this call, so the spoken reply only repeats what just happened."""
    fresh = [s for s in run.steps[since:] if s.status in {StepStatus.OK, StepStatus.FAILED}]
    said = " ".join(s.speech.strip() for s in fresh if s.speech.strip())
    needs = run.status == RunStatus.AWAITING_CONFIRMATION and run.pending is not None
    if needs:
        prompt = run.steps[run.cursor].speech
        speech = f"{said} {prompt}".strip()
    elif run.status == RunStatus.COMPLETED:
        speech = said or "Done."
    elif run.status == RunStatus.FAILED:
        speech = f"{said} I stopped there.".strip() if said else (
            f"That didn't work: {run.error}. I stopped there.")
    elif run.status == RunStatus.REJECTED:
        speech = "Part of that isn't something I can do, so I didn't do any of it."
    elif run.status == RunStatus.CANCELLED:
        speech = ("That confirmation expired, so I stopped." if "expired" in run.error
                  else "Okay, I stopped.")
    else:
        speech = said
    lines = [
        f"{_STEP_MARK.get(s.status, '·')} {s.index + 1}. "
        f"{s.decision.intent.value.replace('_', ' ')}"
        + (f": {s.speech.strip()}" if s.speech.strip() else "")
        for s in run.steps
    ]
    if run.status == RunStatus.REJECTED:
        lines = [f"Not started: {run.error}"]
    last_output = next((s.message for s in reversed(fresh) if s.message), "")
    text = "\n".join(lines) + (f"\n\n{last_output}" if last_output else "")
    data: dict[str, Any] = run.to_dict()
    return AssistantResponse(
        text=text,
        state=str(run.status),
        needs_confirmation=needs,
        speech=speech,
        confirmation_id=run.pending.id if needs and run.pending else "",
        expires_in=run.pending.expires_in if needs and run.pending else 0.0,
        run=data,
    )

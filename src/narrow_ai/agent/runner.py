"""Execute a Plan step by step through the ordinary ToolExecutor.

Nothing here can execute a tool directly: every step is handed to
``ToolExecutor.execute``, which runs the permission gate, parameter validation,
the tool's precheck and confirmation exactly as for a single request. The
runner only adds ordering, pausing, budgets and bookkeeping.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from narrow_ai.agent.events import EventBus, EventType, RunStore
from narrow_ai.agent.plan import Plan, PlanRejected, validate_plan
from narrow_ai.agent.planner import Planner
from narrow_ai.intents.classifier import IntentParseError
from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import IntentDecision
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.security.audit import AuditLog
from narrow_ai.security.confirmation import Clock, ConfirmCheck, PendingAction
from narrow_ai.settings import Settings
from narrow_ai.tools.executor import ToolExecutor
from narrow_ai.tools.registry import ToolRegistry, ToolState
from narrow_ai.tools.result import ToolResult

logger = logging.getLogger(__name__)

SPEECH_LIMIT = 200  # what is kept/persisted per step: the short summary, not tool output


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"  # the plan itself was refused; nothing ran
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.REJECTED, RunStatus.CANCELLED}


@dataclass
class StepState:
    index: int
    decision: IntentDecision
    status: StepStatus = StepStatus.PENDING
    ok: bool = False
    message: str = ""  # full tool output: in memory / API response only, never persisted
    speech: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self, *, with_output: bool = True) -> dict[str, Any]:
        d = {
            "index": self.index,
            "intent": self.decision.intent.value,
            "parameters": self.decision.parameters,
            "status": str(self.status),
            "ok": self.ok,
            "speech": self.speech[:SPEECH_LIMIT],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if with_output:
            d["message"] = self.message
        return d


@dataclass
class Run:
    id: str
    goal: str
    status: RunStatus
    steps: list[StepState] = field(default_factory=list)
    cursor: int = 0
    active_sec: float = 0.0
    error: str = ""
    pending: PendingAction | None = None

    def to_dict(self, *, include_confirmation: bool = False, with_output: bool = True) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "goal": self.goal,
            "status": str(self.status),
            "steps": [s.to_dict(with_output=with_output) for s in self.steps],
            "error": self.error,
            "awaiting_step": (
                self.cursor if self.status == RunStatus.AWAITING_CONFIRMATION else None
            ),
        }
        if include_confirmation and self.pending is not None:
            d["confirmation_id"] = self.pending.id
            d["expires_in"] = round(self.pending.expires_in, 1)
        return d


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        registry: IntentRegistry,
        *,
        preferences: PreferenceStore | None = None,
        answerer: Callable[[str], str] | None = None,
        explainer: Callable[[str], str] | None = None,
        tool_state: ToolState | None = None,
        tools: ToolRegistry | None = None,
        store: RunStore | None = None,
        bus: EventBus | None = None,
        audit: AuditLog | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.preferences = preferences
        self.answerer = answerer
        self.explainer = explainer
        self.tool_state = tool_state if tool_state is not None else ToolState()
        self.tools = tools
        self.store = store
        self.bus = bus or EventBus()
        self.audit = audit or AuditLog()
        self.clock = clock
        self._runs: OrderedDict[str, Run] = OrderedDict()
        if store is not None:
            self.bus.subscribe(store.save_event)

    # ------------------------------------------------------------------ public
    def start(self, text: str, planner: Planner) -> Run:
        run = Run(id=secrets.token_urlsafe(9), goal=text.strip()[:2000], status=RunStatus.RUNNING)
        self._remember(run)
        try:
            plan = validate_plan(
                planner.plan(text), self.registry, max_steps=self.settings.agent.max_steps
            )
        except (PlanRejected, IntentParseError, ValidationError, ValueError) as exc:
            run.status = RunStatus.REJECTED
            run.error = str(exc).splitlines()[0][:300]
            self.audit.record("agent_plan", False, f"plan rejected: {run.error}")
            self._finish(run)
            return run
        run.steps = [StepState(i, s.decision) for i, s in enumerate(plan.steps)]
        self._emit(run, EventType.RUN_STARTED, steps=[s.decision.intent.value for s in plan.steps])
        self._advance(run)
        return run

    def confirm(self, run_id: str, confirmation_id: str | None) -> Run | None:
        run = self.get(run_id)
        if run is None:
            return None
        pending = run.pending
        if run.status != RunStatus.AWAITING_CONFIRMATION or pending is None:
            return run
        check = pending.check(confirmation_id, require_id=True)
        run.pending = None  # single use, whatever the outcome
        step = run.steps[run.cursor]
        self.audit.record(step.decision.intent.value, check == ConfirmCheck.OK,
                          f"run {run.id} step {step.index} confirmation {check}")
        if check != ConfirmCheck.OK:
            step.status = StepStatus.CANCELLED
            run.status = RunStatus.CANCELLED
            run.error = (
                "confirmation expired" if check == ConfirmCheck.EXPIRED
                else "confirmation did not match"
            )
            self._skip_rest(run, run.cursor + 1)
            self._finish(run)
            return run
        run.status = RunStatus.RUNNING
        if not self._run_step(run, step, confirmed=True):
            return run
        run.cursor += 1
        self._advance(run)
        return run

    def cancel(self, run_id: str) -> Run | None:
        run = self.get(run_id)
        if run is None or run.status in TERMINAL:
            return run
        run.pending = None
        run.status = RunStatus.CANCELLED
        run.error = "cancelled by user"
        if run.cursor < len(run.steps):
            run.steps[run.cursor].status = StepStatus.CANCELLED
        self._skip_rest(run, run.cursor + 1)
        self._finish(run)
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.bus.history(run_id)]

    # ------------------------------------------------------------------ internals
    def _executor(self, *, confirmed: bool) -> ToolExecutor:
        return ToolExecutor(
            self.settings,
            self.registry,
            self.preferences,
            confirmed=confirmed,
            answerer=self.answerer,
            tools=self.tools,
            explainer=self.explainer,
            state=self.tool_state,
        )

    def _advance(self, run: Run) -> None:
        budget = self.settings.agent.max_wall_sec
        while run.cursor < len(run.steps):
            step = run.steps[run.cursor]
            if run.active_sec >= budget:
                run.status = RunStatus.FAILED
                run.error = f"time budget of {budget:g}s exceeded"
                self._skip_rest(run, run.cursor)
                self._finish(run)
                return
            if not self._run_step(run, step, confirmed=False):
                return
            run.cursor += 1
        run.status = RunStatus.COMPLETED
        self._finish(run)

    def _run_step(self, run: Run, step: StepState, *, confirmed: bool) -> bool:
        """Run one step; return True to continue with the next step."""
        step.status = StepStatus.RUNNING
        self._emit(run, EventType.STEP_STARTED, index=step.index, intent=step.decision.intent.value)
        t0 = self.clock()
        result: ToolResult = self._executor(confirmed=confirmed).execute(step.decision)
        dt = max(0.0, self.clock() - t0)
        run.active_sec += dt
        step.elapsed_ms = dt * 1000
        if result.needs_confirmation and not confirmed:
            run.pending = PendingAction(
                step.decision,
                ttl_sec=float(self.settings.security.confirmation_ttl_sec),
                clock=self.clock,
            )
            step.status = StepStatus.AWAITING_CONFIRMATION
            step.speech = result.speech
            run.status = RunStatus.AWAITING_CONFIRMATION
            self.audit.record(step.decision.intent.value, False,
                              f"run {run.id} step {step.index} awaiting confirmation")
            self._emit(run, EventType.AWAITING_CONFIRMATION, index=step.index,
                       intent=step.decision.intent.value, prompt=result.speech[:SPEECH_LIMIT])
            self._persist(run)
            return False
        step.ok = result.ok
        step.message = result.message
        step.speech = result.speech or result.message[:SPEECH_LIMIT]
        step.status = StepStatus.OK if result.ok else StepStatus.FAILED
        self.audit.record(step.decision.intent.value, result.ok,
                          f"run {run.id} step {step.index}: {result.message[:120]}")
        self._emit(run, EventType.STEP_RESULT, index=step.index,
                   intent=step.decision.intent.value, ok=result.ok,
                   speech=step.speech[:SPEECH_LIMIT])
        if not result.ok:  # fail closed: stop at the first failed step
            run.status = RunStatus.FAILED
            run.error = f"step {step.index} ({step.decision.intent.value}) failed"
            self._skip_rest(run, step.index + 1)
            self._finish(run)
            return False
        return True

    def _skip_rest(self, run: Run, start: int) -> None:
        for s in run.steps[start:]:
            if s.status == StepStatus.PENDING:
                s.status = StepStatus.SKIPPED

    def _finish(self, run: Run) -> None:
        self._emit(run, EventType.RUN_FINISHED, status=str(run.status), error=run.error)
        self._persist(run)

    def _emit(self, run: Run, type: EventType, **data: Any) -> None:
        self.bus.publish(run.id, type, **data)

    def _persist(self, run: Run) -> None:
        if self.store is not None:
            try:
                self.store.save_run(run.to_dict(with_output=False))
            except Exception:  # noqa: BLE001 - persistence must never break a run
                logger.exception("could not persist run %s", run.id)

    def _remember(self, run: Run) -> None:
        self._runs[run.id] = run
        while len(self._runs) > self.settings.agent.max_runs_in_memory:
            old_id, _ = self._runs.popitem(last=False)
            self.bus.forget(old_id)


def plan_from_decisions(goal: str, decisions: list[IntentDecision]) -> Plan:
    """Helper for tests and scripted callers."""
    from narrow_ai.agent.plan import PlanStep

    return Plan(goal=goal, steps=[PlanStep(decision=d) for d in decisions])

"""Main text/voice orchestration loop."""

from __future__ import annotations

import json
import logging
import time

from narrow_ai.assistant.conversation import AssistantState
from narrow_ai.assistant.response import AssistantResponse
from narrow_ai.intents.classifier import IntentParseError, heuristic_intent
from narrow_ai.intents.registry import IntentRegistry, default_registry
from narrow_ai.intents.schemas import IntentDecision
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.memory.store import ConversationStore
from narrow_ai.models.base import GenerationResult, ModelBackend
from narrow_ai.models.output_parser import parse_model_output
from narrow_ai.models.prompt_templates import (
    ANSWER_SYSTEM,
    EXPLAIN_TESTS_SYSTEM,
    INTENT_PREFILL,
    build_system_prompt,
    implemented_intent_names,
)
from narrow_ai.security.audit import AuditLog
from narrow_ai.security.confirmation import Clock, ConfirmCheck, PendingAction
from narrow_ai.settings import Settings
from narrow_ai.tools.executor import ToolExecutor
from narrow_ai.tools.registry import ToolState

logger = logging.getLogger(__name__)

YES = {"yes", "y", "yeah", "yep", "sure", "confirm", "do it", "ok", "okay", "yes please"}
NO = {"no", "n", "nope", "cancel", "stop", "never mind", "nevermind"}


def _norm(text: str) -> str:
    return text.strip().lower().strip(".!?, ")


def system_prompt_for(settings: Settings, registry: IntentRegistry) -> str:
    return build_system_prompt(
        settings.assistant.system_prompt,
        registry,
        approved_apps=settings.tools.approved_apps,
        few_shot=settings.model.few_shot,
        test_command_ids=list(settings.tools.test_commands),
    )


class IntentEngine:
    """Utterance → validated IntentDecision (shared by the app and the evaluator)."""

    def __init__(self, settings: Settings, backend: ModelBackend, registry: IntentRegistry):
        self.settings = settings
        self.backend = backend
        self.registry = registry
        self.system_prompt = system_prompt_for(settings, registry)
        self.allowed = implemented_intent_names(registry)
        self.constraint = settings.model.constraint_mode()
        self.schema: dict | None = None
        if self.constraint == "json":
            from narrow_ai.models.grammar import decision_json_schema

            self.schema = decision_json_schema(registry)
            if backend.name == "mlx":  # fail at startup, never silently downgrade
                from narrow_ai.models.json_constraint import require_llguidance

                require_llguidance()

    def decide(
        self, user_text: str, history: list[dict[str, str]] | None = None
    ) -> tuple[IntentDecision, GenerationResult | None, str]:
        """Return (decision, generation, source) where source is model|retry|heuristic."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += history or []
        messages.append({"role": "user", "content": user_text})
        kwargs = {
            "max_tokens": self.settings.model.max_tokens,
            "temperature": self.settings.model.temperature,
            "prefill": INTENT_PREFILL,
            "constrain_intents": self.allowed if self.constraint == "intent" else None,
            "constraint_schema": self.schema,
        }
        gen = self.backend.generate(messages, **kwargs)
        try:
            return parse_model_output(gen.text), gen, "model"
        except IntentParseError as exc:
            logger.warning("parse failed (%s); retrying once", exc)
        retry_msgs = messages + [
            {"role": "assistant", "content": gen.text},
            {
                "role": "user",
                "content": "That was not a valid JSON intent. Reply with only the JSON object.",
            },
        ]
        gen2 = self.backend.generate(retry_msgs, **kwargs)
        try:
            return parse_model_output(gen2.text), gen2, "retry"
        except IntentParseError:
            decision = heuristic_intent(user_text)
            logger.warning("falling back to heuristic intent: %s", decision.intent)
            return decision, gen2, "heuristic"


class AssistantOrchestrator:
    def __init__(
        self,
        settings: Settings,
        backend: ModelBackend,
        *,
        registry: IntentRegistry | None = None,
        preferences: PreferenceStore | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.registry = registry or default_registry(settings)
        self.preferences = preferences
        if preferences is None and settings.memory.enabled:
            self.preferences = PreferenceStore(
                settings.db_path(),
                max_preferences=settings.memory.max_preferences,
            )
        self.engine = IntentEngine(settings, backend, self.registry)
        # History holds only (user, assistant-JSON) pairs: short and format-consistent.
        self.conversation = ConversationStore(max_turns=settings.model.max_history_turns)
        self.audit = AuditLog()
        self.state = AssistantState.IDLE
        self.clock = clock
        self._pending: PendingAction | None = None
        self.tool_state = ToolState()  # e.g. the last test run, for "why did it fail?"

    @property
    def system_prompt(self) -> str:
        return self.engine.system_prompt

    def answer(self, question: str) -> str:
        """Short spoken-style answer from the local model (no tools, no execution)."""
        if self.backend.name == "fake":
            return "The offline fake backend can't answer questions; start R2 with the MLX model."
        gen = self.backend.generate(
            [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": question}],
            max_tokens=180,
            temperature=0.0,
        )
        text = gen.text.replace("<think>", "").replace("</think>", "").strip()
        text = text.replace("**", "").replace("`", "")
        if len(text) > 700:  # keep it speakable; cut at a sentence boundary
            cut = text[:700].rfind(". ")
            text = text[: cut + 1] if cut > 200 else text[:700] + "…"
        return text

    def explain_tests(self, excerpt: str) -> str:
        """Second, text-only model call: explain a failing test run. The excerpt is
        untrusted (it is the project's output); the reply is displayed and spoken,
        never parsed into an action."""
        if self.backend.name == "fake":
            return ""  # caller falls back to the deterministic summary
        gen = self.backend.generate(
            [
                {"role": "system", "content": EXPLAIN_TESTS_SYSTEM},
                {"role": "user", "content": f"<output>\n{excerpt}\n</output>"},
            ],
            max_tokens=160,
            temperature=0.0,
        )
        text = gen.text.replace("<think>", "").replace("</think>", "").strip()
        return text.replace("**", "").replace("`", "")[:700]

    def warmup(self) -> None:
        """Precompute the KV prefix cache and compile the decode path (incl. the
        intent constraint's vocabulary index) so the first real request is fast."""
        self.backend.warmup(self.engine.system_prompt)
        if self.backend.name == "mlx":
            self.backend.warmup(ANSWER_SYSTEM)  # second cached prefix for Q&A
            if self.settings.tools.run_tests_enabled():
                self.backend.warmup(EXPLAIN_TESTS_SYSTEM)  # third: test explanations
        self.engine.decide("what can you do")  # not recorded in history
        if self.backend.name == "mlx":
            st = self.backend.cache_stats()
            st.hits = st.misses = 0

    @property
    def pending(self) -> IntentDecision | None:
        return self._pending.decision if self._pending else None

    @property
    def pending_action(self) -> PendingAction | None:
        return self._pending

    def confirm(
        self, confirmation_id: str | None = None, *, require_id: bool = True
    ) -> AssistantResponse:
        """Approve the parked action. Fail closed: any invalid attempt cancels it.

        ``require_id=False`` is only for trusted in-process front ends (terminal UI,
        notebooks) where the person typing *is* the one who saw the prompt. The
        HTTP API always requires the single-use id issued with the prompt.
        """
        pending = self._pending
        if pending is None:
            return AssistantResponse(text="Nothing to confirm.", speech="Nothing to confirm.")
        check = pending.check(confirmation_id, require_id=require_id)
        self._pending = None  # single use, whatever the outcome
        intent = pending.decision.intent.value
        self.audit.record(intent, check == ConfirmCheck.OK, f"confirmation {check}")
        if check != ConfirmCheck.OK:
            self.state = AssistantState.IDLE
            msg = (
                "That confirmation expired, so I cancelled it. Please ask again."
                if check == ConfirmCheck.EXPIRED
                else "That confirmation didn't match the pending action, so I cancelled it."
            )
            logger.warning("confirmation rejected (%s) for %s", check, intent)
            return AssistantResponse(
                text=msg, state=AssistantState.IDLE, speech="Okay, I cancelled that."
            )
        return self._execute(pending.decision, confirmed=True)

    def handle_text(
        self,
        user_text: str,
        *,
        confirmation_id: str | None = None,
        require_confirmation_id: bool = False,
    ) -> AssistantResponse:
        user_text = user_text.strip()
        lowered = _norm(user_text)

        if self._pending is not None:
            if lowered in YES:
                return self.confirm(confirmation_id, require_id=require_confirmation_id)
            if lowered in NO:
                self.audit.record(self._pending.decision.intent.value, False, "cancelled")
                self.cancel()
                return AssistantResponse(
                    text="Okay, cancelled.", state=AssistantState.IDLE, speech="Okay, cancelled."
                )
            # Anything else: drop the pending action and treat as a new request.
            self.audit.record(self._pending.decision.intent.value, False, "superseded")
            self._pending = None

        if not user_text:
            return AssistantResponse(text="Empty request.", state=AssistantState.ERROR)

        self.state = AssistantState.THINKING
        t0 = time.perf_counter()
        try:
            decision, gen, source = self.engine.decide(user_text, self.conversation.as_chat())
        except Exception as exc:  # noqa: BLE001
            self.state = AssistantState.ERROR
            logger.exception("generation failed")
            return AssistantResponse(text=f"Model error: {exc}", state=AssistantState.ERROR)
        think_ms = (time.perf_counter() - t0) * 1000

        self.conversation.add("user", user_text)
        self.conversation.add(
            "assistant",
            json.dumps(
                {"intent": decision.intent.value, "parameters": decision.parameters},
                ensure_ascii=False,
            ),
        )
        resp = self._execute(decision, confirmed=False)
        resp.metrics = gen.metrics() if gen else {}
        resp.metrics["think_ms"] = round(think_ms, 1)
        resp.metrics["source"] = source
        return resp

    def _execute(self, decision: IntentDecision, *, confirmed: bool) -> AssistantResponse:
        self.state = AssistantState.EXECUTING
        executor = ToolExecutor(
            self.settings,
            self.registry,
            self.preferences,
            confirmed=confirmed,
            answerer=self.answer,
            explainer=self.explain_tests,
            state=self.tool_state,
        )
        result = executor.execute(decision)
        self.audit.record(decision.intent.value, result.ok, result.message)

        pending: PendingAction | None = None
        if result.needs_confirmation:
            pending = PendingAction(
                decision,
                ttl_sec=float(self.settings.security.confirmation_ttl_sec),
                clock=self.clock,
            )
            self._pending = pending
            self.state = AssistantState.AWAITING_CONFIRMATION
        else:
            self.state = AssistantState.IDLE
        return AssistantResponse(
            text=result.message,
            decision=decision,
            tool_result=result,
            state=self.state,
            needs_confirmation=result.needs_confirmation,
            speech=result.speech,
            confirmation_id=pending.id if pending else "",
            expires_in=pending.ttl_sec if pending else 0.0,
        )

    def cancel(self) -> None:
        self._pending = None
        self.state = AssistantState.IDLE

    def reset(self) -> None:
        self.cancel()
        self.conversation.clear()

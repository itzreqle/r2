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
    INTENT_PREFILL,
    build_system_prompt,
    implemented_intent_names,
)
from narrow_ai.security.audit import AuditLog
from narrow_ai.settings import Settings
from narrow_ai.tools.executor import ToolExecutor

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
    )


class IntentEngine:
    """Utterance → validated IntentDecision (shared by the app and the evaluator)."""

    def __init__(self, settings: Settings, backend: ModelBackend, registry: IntentRegistry):
        self.settings = settings
        self.backend = backend
        self.registry = registry
        self.system_prompt = system_prompt_for(settings, registry)
        self.allowed = implemented_intent_names(registry)

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
            "constrain_intents": self.allowed if self.settings.model.constrained else None,
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
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.registry = registry or default_registry()
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
        self._pending: IntentDecision | None = None

    @property
    def system_prompt(self) -> str:
        return self.engine.system_prompt

    def warmup(self) -> None:
        """Precompute the KV prefix cache so the first request is fast."""
        self.backend.warmup(self.engine.system_prompt)

    @property
    def pending(self) -> IntentDecision | None:
        return self._pending

    def handle_text(self, user_text: str, *, confirmed: bool = False) -> AssistantResponse:
        user_text = user_text.strip()
        lowered = _norm(user_text)

        if self._pending is not None:
            if confirmed or lowered in YES:
                decision = self._pending
                self._pending = None
                return self._execute(decision, confirmed=True)
            if lowered in NO:
                self.cancel()
                return AssistantResponse(
                    text="Okay, cancelled.", state=AssistantState.IDLE, speech="Okay, cancelled."
                )
            # Anything else: drop the pending action and treat as a new request.
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
        resp = self._execute(decision, confirmed=confirmed)
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
        )
        result = executor.execute(decision)
        self.audit.record(decision.intent.value, result.ok, result.message)

        if result.needs_confirmation:
            self._pending = decision
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
        )

    def cancel(self) -> None:
        self._pending = None
        self.state = AssistantState.IDLE

    def reset(self) -> None:
        self.cancel()
        self.conversation.clear()

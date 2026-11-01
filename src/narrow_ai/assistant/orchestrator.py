"""Main text/voice orchestration loop."""

from __future__ import annotations

import logging

from narrow_ai.assistant.conversation import AssistantState
from narrow_ai.assistant.response import AssistantResponse
from narrow_ai.intents.classifier import IntentParseError, heuristic_intent
from narrow_ai.intents.registry import IntentRegistry, default_registry
from narrow_ai.intents.schemas import IntentDecision
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.memory.store import ConversationStore
from narrow_ai.models.base import ModelBackend
from narrow_ai.models.output_parser import parse_model_output
from narrow_ai.models.prompt_templates import build_system_prompt
from narrow_ai.security.audit import AuditLog
from narrow_ai.settings import Settings
from narrow_ai.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


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
        self.conversation = ConversationStore(max_turns=settings.model.max_history_turns)
        system = build_system_prompt(settings.assistant.system_prompt, self.registry)
        self.conversation.add("system", system)
        self.audit = AuditLog()
        self.state = AssistantState.IDLE
        self._pending: IntentDecision | None = None

    def handle_text(self, user_text: str, *, confirmed: bool = False) -> AssistantResponse:
        user_text = user_text.strip()
        lowered = user_text.lower()

        if self._pending is not None:
            if confirmed or lowered in {"yes", "y", "confirm"}:
                decision = self._pending
                self._pending = None
                return self._execute(decision, confirmed=True)
            if lowered in {"no", "n", "cancel"}:
                self.cancel()
                return AssistantResponse(text="Cancelled pending action.", state=AssistantState.IDLE)

        if not user_text:
            return AssistantResponse(text="Empty request.", state=AssistantState.ERROR)

        self.state = AssistantState.THINKING
        self.conversation.add("user", user_text)

        try:
            gen = self.backend.generate(
                self.conversation.as_chat(),
                max_tokens=self.settings.model.max_tokens,
                temperature=self.settings.model.temperature,
            )
            try:
                decision = parse_model_output(gen.text)
            except IntentParseError:
                logger.warning("parse failed; retrying once with correction prompt")
                self.conversation.add(
                    "user",
                    "Your previous reply was not valid JSON. "
                    "Reply with only a JSON object for the user's last request.",
                )
                gen2 = self.backend.generate(self.conversation.as_chat())
                try:
                    decision = parse_model_output(gen2.text)
                except IntentParseError:
                    decision = heuristic_intent(user_text)
                    logger.warning("falling back to heuristic intent: %s", decision.intent)
        except Exception as exc:  # noqa: BLE001
            self.state = AssistantState.ERROR
            logger.exception("generation failed")
            return AssistantResponse(text=f"Model error: {exc}", state=AssistantState.ERROR)

        return self._execute(decision, confirmed=confirmed)

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
            msg = (
                f"Confirm `{decision.intent}` with {decision.parameters}? "
                "Reply `yes` to proceed or `no` to cancel."
            )
            self.conversation.add("assistant", msg)
            return AssistantResponse(
                text=msg,
                decision=decision,
                tool_result=result,
                state=self.state,
                needs_confirmation=True,
            )

        self.conversation.add("assistant", result.message)
        self.state = AssistantState.IDLE
        return AssistantResponse(
            text=result.message,
            decision=decision,
            tool_result=result,
            state=self.state,
        )

    def cancel(self) -> None:
        self._pending = None
        self.state = AssistantState.IDLE

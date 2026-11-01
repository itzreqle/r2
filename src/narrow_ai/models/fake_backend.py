"""Deterministic backend for tests and offline text mode (keyword heuristics)."""

from __future__ import annotations

import json
import time

from narrow_ai.intents.classifier import heuristic_intent
from narrow_ai.models.base import GenerationResult, ModelBackend


class FakeModelBackend(ModelBackend):
    name = "fake"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prefill: str = "",
        constrain_intents: list[str] | None = None,
    ) -> GenerationResult:
        _ = max_tokens, temperature, prefill, constrain_intents
        t0 = time.perf_counter()
        user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user = msg.get("content", "")
                break
        decision = heuristic_intent(user)
        text = json.dumps(
            {"intent": decision.intent.value, "parameters": decision.parameters},
            ensure_ascii=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            text=text,
            generate_time_ms=elapsed,
            ttft_ms=elapsed,
            backend=self.name,
            meta={"heuristic": True},
        )

"""Run baseline / adapter evaluation without side effects."""

from __future__ import annotations

import json
from pathlib import Path

from narrow_ai.evaluation.dataset import load_eval_cases
from narrow_ai.evaluation.metrics import IntentMetrics, score_intents
from narrow_ai.intents.classifier import IntentParseError
from narrow_ai.intents.schemas import IntentName
from narrow_ai.models.base import ModelBackend
from narrow_ai.models.output_parser import parse_model_output
from narrow_ai.models.prompt_templates import build_system_prompt
from narrow_ai.intents.registry import default_registry
from narrow_ai.settings import Settings


class EvaluationRunner:
    def __init__(self, settings: Settings, backend: ModelBackend) -> None:
        self.settings = settings
        self.backend = backend
        self.registry = default_registry()

    def run(self, eval_path: Path) -> IntentMetrics:
        cases = load_eval_cases(eval_path)
        system = build_system_prompt(self.settings.assistant.system_prompt, self.registry)
        gold: list[IntentName] = []
        pred: list[IntentName | None] = []
        for case in cases:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": case.utterance},
            ]
            result = self.backend.generate(messages)
            try:
                decision = parse_model_output(result.text)
                pred.append(decision.intent)
            except IntentParseError:
                pred.append(None)
            gold.append(case.expected_intent)
        return score_intents(gold, pred)

    @staticmethod
    def write_report(metrics: IntentMetrics, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "accuracy": metrics.accuracy,
            "n": metrics.n,
            "malformed_rate": metrics.malformed_rate,
            "per_intent_f1": metrics.per_intent_f1,
            "confusion": metrics.confusion,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

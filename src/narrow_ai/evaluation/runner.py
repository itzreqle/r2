"""Run baseline / adapter evaluation without side effects (tools are NOT executed)."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from narrow_ai.assistant.orchestrator import IntentEngine
from narrow_ai.evaluation.dataset import EvalCase, load_eval_cases
from narrow_ai.evaluation.metrics import IntentMetrics, score_intents
from narrow_ai.intents.registry import default_registry
from narrow_ai.intents.schemas import IntentName
from narrow_ai.models.base import ModelBackend
from narrow_ai.settings import Settings


def params_match(expected: dict[str, Any], got: dict[str, Any]) -> bool:
    """Every expected key must be present with a case-insensitive equal value."""
    for k, v in expected.items():
        if str(got.get(k, "")).strip().lower() != str(v).strip().lower():
            return False
    return True


@dataclass
class CaseResult:
    id: str
    utterance: str
    expected: str
    predicted: str | None
    parameters: dict[str, Any]
    intent_ok: bool
    params_ok: bool
    source: str
    latency_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    metrics: IntentMetrics
    param_accuracy: float
    fallback_rate: float
    latency_ms_mean: float
    latency_ms_p50: float
    ttft_ms_mean: float
    cases: list[CaseResult]
    config: dict[str, Any]


class EvaluationRunner:
    def __init__(self, settings: Settings, backend: ModelBackend) -> None:
        self.settings = settings
        self.backend = backend
        self.registry = default_registry()
        self.engine = IntentEngine(settings, backend, self.registry)

    def run_cases(self, cases: list[EvalCase], progress: Any = None) -> EvalReport:
        results: list[CaseResult] = []
        for i, case in enumerate(cases):
            t0 = time.perf_counter()
            decision, gen, source = self.engine.decide(case.utterance)
            latency = (time.perf_counter() - t0) * 1000
            pred = decision.intent if source != "heuristic" else None
            results.append(
                CaseResult(
                    id=case.id,
                    utterance=case.utterance,
                    expected=case.expected_intent.value,
                    predicted=decision.intent.value,
                    parameters=decision.parameters,
                    intent_ok=pred == case.expected_intent,
                    params_ok=pred == case.expected_intent
                    and params_match(case.expected_parameters, decision.parameters),
                    source=source,
                    latency_ms=latency,
                    metrics=gen.metrics() if gen else {},
                )
            )
            if progress:
                progress(i + 1, len(cases), results[-1])
        gold = [IntentName(r.expected) for r in results]
        pred_l = [IntentName(r.predicted) if r.source != "heuristic" else None for r in results]
        lat = [r.latency_ms for r in results] or [0.0]
        ttft = [r.metrics.get("ttft_ms", 0.0) for r in results] or [0.0]
        n = len(results) or 1
        return EvalReport(
            metrics=score_intents(gold, pred_l),
            param_accuracy=sum(r.params_ok for r in results) / n,
            fallback_rate=sum(r.source == "heuristic" for r in results) / n,
            latency_ms_mean=statistics.mean(lat),
            latency_ms_p50=statistics.median(lat),
            ttft_ms_mean=statistics.mean(ttft),
            cases=results,
            config={
                "backend": self.backend.name,
                "model_id": self.settings.model.model_id,
                "cache_mode": self.settings.model.cache_mode,
                "constrained": self.settings.model.constrained,
                "few_shot": self.settings.model.few_shot,
                "adapter": self.settings.model.adapter_path,
            },
        )

    def run(self, eval_path: Path) -> IntentMetrics:
        return self.run_cases(load_eval_cases(eval_path)).metrics

    @staticmethod
    def write_report(report: EvalReport | IntentMetrics, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(report, IntentMetrics):
            payload: dict[str, Any] = asdict(report)
        else:
            payload = {
                "accuracy": report.metrics.accuracy,
                "param_accuracy": report.param_accuracy,
                "n": report.metrics.n,
                "malformed_rate": report.metrics.malformed_rate,
                "fallback_rate": report.fallback_rate,
                "latency_ms_mean": report.latency_ms_mean,
                "latency_ms_p50": report.latency_ms_p50,
                "ttft_ms_mean": report.ttft_ms_mean,
                "per_intent_f1": report.metrics.per_intent_f1,
                "confusion": report.metrics.confusion,
                "config": report.config,
                "cases": [asdict(c) for c in report.cases],
            }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

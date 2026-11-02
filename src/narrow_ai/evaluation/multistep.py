"""Planner evaluation on data/eval_multistep.jsonl. Plans are built and validated,
never executed: no tool runs during evaluation.

A case's ``expected`` is either a list of steps ``{"intent", "parameters"}`` (only the
listed parameters are compared, case-insensitively) or the string ``"rejected"``,
meaning a correct planner must refuse the whole request.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from narrow_ai.agent.plan import MAX_STEPS, PlanRejected, validate_plan
from narrow_ai.agent.planner import Planner
from narrow_ai.evaluation.runner import params_match
from narrow_ai.intents.classifier import IntentParseError
from narrow_ai.intents.registry import IntentRegistry

REJECTED = "rejected"


@dataclass
class MultiStepCase:
    id: str
    utterance: str
    category: str
    expected: list[dict[str, Any]] | str


@dataclass
class MultiStepResult:
    id: str
    category: str
    utterance: str
    expected: list[dict[str, Any]] | str
    got: list[dict[str, Any]] | str
    plan_ok: bool
    intents_ok: bool
    count_ok: bool
    latency_ms: float
    error: str = ""


@dataclass
class MultiStepReport:
    planner: str
    n: int
    plan_accuracy: float
    plan_ci95: tuple[float, float]
    intent_sequence_accuracy: float
    step_count_accuracy: float
    reject_recall: float  # of the requests that must be refused, how many were
    false_reject_rate: float  # of the requests that should run, how many were refused
    latency_ms_mean: float
    latency_ms_p50: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    cases: list[MultiStepResult] = field(default_factory=list)


def load_cases(path: Path) -> list[MultiStepCase]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(MultiStepCase(d["id"], d["utterance"], d.get("category", ""), d["expected"]))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(
    expected: list[dict[str, Any]] | str, got: list[dict[str, Any]] | str
) -> tuple[bool, bool, bool]:
    """(plan_ok, intents_ok, count_ok)."""
    if expected == REJECTED or got == REJECTED:
        same = expected == got
        return same, same, same
    assert isinstance(expected, list) and isinstance(got, list)
    count_ok = len(expected) == len(got)
    intents_ok = count_ok and all(
        e["intent"] == g["intent"] for e, g in zip(expected, got, strict=True)
    )
    plan_ok = intents_ok and all(
        params_match(e.get("parameters", {}), g.get("parameters", {}))
        for e, g in zip(expected, got, strict=True)
    )
    return plan_ok, intents_ok, count_ok


def run_planner(
    name: str,
    planner: Planner,
    registry: IntentRegistry,
    cases: list[MultiStepCase],
    *,
    max_steps: int = MAX_STEPS,
    progress: Any = None,
) -> MultiStepReport:
    results: list[MultiStepResult] = []
    for case in cases:
        t0 = time.perf_counter()
        err = ""
        try:
            plan = validate_plan(planner.plan(case.utterance), registry, max_steps=max_steps)
            got: list[dict[str, Any]] | str = [
                {"intent": s.decision.intent.value, "parameters": s.decision.parameters}
                for s in plan.steps
            ]
        except (PlanRejected, IntentParseError, ValidationError, ValueError) as exc:
            got, err = REJECTED, str(exc).splitlines()[0][:200]
        ms = (time.perf_counter() - t0) * 1000
        plan_ok, intents_ok, count_ok = score(case.expected, got)
        results.append(MultiStepResult(case.id, case.category, case.utterance, case.expected, got,
                                       plan_ok, intents_ok, count_ok, round(ms, 1), err))
        if progress:
            progress(results[-1])
    return summarize(name, results)


def summarize(name: str, results: list[MultiStepResult]) -> MultiStepReport:
    n = len(results)
    k = sum(r.plan_ok for r in results)
    must_reject = [r for r in results if r.expected == REJECTED]
    should_run = [r for r in results if r.expected != REJECTED]
    lat = [r.latency_ms for r in results] or [0.0]
    cats: dict[str, list[MultiStepResult]] = defaultdict(list)
    for r in results:
        cats[r.category].append(r)
    return MultiStepReport(
        planner=name,
        n=n,
        plan_accuracy=k / n if n else 0.0,
        plan_ci95=wilson(k, n),
        intent_sequence_accuracy=sum(r.intents_ok for r in results) / n if n else 0.0,
        step_count_accuracy=sum(r.count_ok for r in results) / n if n else 0.0,
        reject_recall=(sum(r.got == REJECTED for r in must_reject) / len(must_reject)
                       if must_reject else 0.0),
        false_reject_rate=(sum(r.got == REJECTED for r in should_run) / len(should_run)
                           if should_run else 0.0),
        latency_ms_mean=statistics.fmean(lat),
        latency_ms_p50=statistics.median(lat),
        by_category={
            c: {"n": len(rs), "plan_accuracy": sum(r.plan_ok for r in rs) / len(rs)}
            for c, rs in sorted(cats.items())
        },
        cases=results,
    )


def report_dict(rep: MultiStepReport) -> dict[str, Any]:
    d = asdict(rep)
    d["plan_ci95"] = [round(x, 3) for x in rep.plan_ci95]
    return d

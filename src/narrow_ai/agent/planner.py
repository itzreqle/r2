"""Turn one request into a Plan. Planning happens once, before any tool runs.

The planner never sees tool output, so nothing a tool returns (file contents,
diffs, answers) can add, remove or change a step. That is the runtime's main
defense against prompt injection.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import ValidationError

from narrow_ai.agent.plan import MAX_STEPS, Plan, PlanRejected, PlanStep
from narrow_ai.intents.classifier import (
    IntentParseError,
    extract_json_object,
    heuristic_intent,
    normalize_decision_data,
)
from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import IntentDecision
from narrow_ai.models.base import ModelBackend
from narrow_ai.models.grammar import decision_json_schema
from narrow_ai.models.prompt_templates import _PARAM_HINTS, _WHEN, PERSONA
from narrow_ai.settings import Settings

_SPLIT = re.compile(
    r"\s*(?:,?\s*\band then\b|,?\s*\bafter that\b|,?\s*\band also\b|,?\s*\bthen\b|;)\s*",
    re.IGNORECASE,
)

PLAN_PREFILL = '{"steps": [{"intent": "'

PLAN_EXAMPLES: list[tuple[str, list[dict]]] = [
    (
        "check my git status and then find the swift files",
        [
            {"intent": "git_status", "parameters": {}},
            {"intent": "search_files", "parameters": {"pattern": "*.swift"}},
        ],
    ),
    (
        "show me the README, then open cursor",
        [
            {"intent": "read_file", "parameters": {"path": "README.md"}},
            {"intent": "open_app", "parameters": {"app_name": "Cursor"}},
        ],
    ),
    (
        "how big is my diff and also what is a rebase",
        [
            {"intent": "git_diff_summary", "parameters": {}},
            {"intent": "answer_question", "parameters": {"question": "what is a rebase"}},
        ],
    ),
    (
        "remember that my shell is zsh then tell me my test runner",
        [
            {"intent": "remember_preference", "parameters": {"key": "shell", "value": "zsh"}},
            {"intent": "recall_preference", "parameters": {"key": "test_runner"}},
        ],
    ),
    ("what's my git status", [{"intent": "git_status", "parameters": {}}]),
    (
        "delete the build folder and then push to main",
        [{"intent": "unsupported", "parameters": {"reason": "Destructive actions not allowed"}}],
    ),
]


class Planner(Protocol):
    def plan(self, text: str) -> Plan: ...


def split_clauses(text: str) -> list[str]:
    return [c.strip(" ,.") for c in _SPLIT.split(text) if c and c.strip(" ,.")]


def _clauses(text: str, max_steps: int = MAX_STEPS) -> list[str]:
    clauses = split_clauses(text) or [text.strip()]
    if len(clauses) > max_steps:
        raise PlanRejected(f"request has {len(clauses)} steps (max {max_steps})")
    return clauses


class HeuristicPlanner:
    """Deterministic, offline: split on "and then" / "then" / ";" and route each clause
    with the keyword heuristic."""

    def plan(self, text: str) -> Plan:
        steps = [PlanStep(decision=heuristic_intent(c)) for c in _clauses(text)]
        return Plan(goal=text.strip()[:2000], steps=steps)


class ClausePlanner:
    """Default planner: deterministic split, then each clause goes through the regular
    single-intent engine (same cached prompt, grammar and strict validation as a normal
    request).

    On Qwen3-1.7B this beat one-shot ``ModelPlanner`` plans, which dropped steps and
    copied few-shot parameters (docs/AGENT_RUNTIME.md). The trade-off: only explicit
    connectors split a request, so "find tests and fixtures" stays one step.
    """

    def __init__(self, engine: Any, max_steps: int = MAX_STEPS) -> None:
        self.engine = engine  # narrow_ai.assistant.orchestrator.IntentEngine
        self.max_steps = max_steps

    def plan(self, text: str) -> Plan:
        steps = [
            PlanStep(decision=self.engine.decide(c)[0]) for c in _clauses(text, self.max_steps)
        ]
        return Plan(goal=text.strip()[:2000], steps=steps)


def planner_system_prompt(registry: IntentRegistry, approved_apps: list[str]) -> str:
    lines = [
        PERSONA,
        "",
        "Split the user's request into an ordered list of at most "
        f"{MAX_STEPS} steps. Reply with exactly ONE JSON object and nothing else:",
        '{"steps": [{"intent": "<intent>", "parameters": {...}}, ...]}',
        "",
        "Each step is one of these intents (parameters) — when to use:",
    ]
    for spec in registry.implemented():
        hint = _PARAM_HINTS.get(spec.name, "{}")
        lines.append(f"- {spec.name.value} {hint} — {_WHEN.get(spec.name, spec.description)}")
    if approved_apps:
        lines.append(f"Approved apps: {', '.join(approved_apps)}.")
    lines += [
        "",
        "Rules: one step per action the user asked for, in the order they asked; never add "
        "steps the user didn't ask for; never invent shell commands; if any part is "
        "unsupported or destructive, reply with a single unsupported step.",
        "",
        "Examples:",
    ]
    for utt, steps in PLAN_EXAMPLES:
        lines.append(f"User: {utt}")
        lines.append(json.dumps({"steps": steps}, ensure_ascii=False))
    return "\n".join(lines)


def plan_json_schema(registry: IntentRegistry, max_steps: int = MAX_STEPS) -> dict:
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_steps,
                "items": decision_json_schema(registry),
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def parse_plan(text: str, goal: str) -> Plan:
    """Strictly parse model output into a Plan (unknown keys/intents → PlanRejected)."""
    try:
        data = extract_json_object(text)
        if set(data) != {"steps"} or not isinstance(data["steps"], list):
            raise PlanRejected('expected exactly {"steps": [...]}')
        steps = []
        for raw in data["steps"]:
            if not isinstance(raw, dict):
                raise PlanRejected("each step must be a JSON object")
            raw = dict(raw)
            raw.setdefault("parameters", {})
            steps.append(PlanStep(decision=IntentDecision.model_validate(
                normalize_decision_data(raw))))
        return Plan(goal=goal[:2000], steps=steps)
    except (IntentParseError, ValidationError, ValueError) as exc:
        if isinstance(exc, PlanRejected):
            raise
        raise PlanRejected(f"could not parse plan: {exc}") from exc


class ModelPlanner:
    """One constrained model call. Its static prompt is one more cached KV prefix.

    Opt-in (``agent.planner = "model"``): with a 1.7B model it can drop steps, so
    :class:`ClausePlanner` is the default.
    """

    def __init__(self, settings: Settings, backend: ModelBackend, registry: IntentRegistry):
        self.settings = settings
        self.backend = backend
        self.max_steps = min(settings.agent.max_steps, MAX_STEPS)
        self.system_prompt = planner_system_prompt(registry, settings.tools.approved_apps)
        self.schema = (
            plan_json_schema(registry, self.max_steps)
            if settings.model.constraint_mode() == "json"
            else None
        )

    def plan(self, text: str) -> Plan:
        gen = self.backend.generate(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=64 * self.max_steps,
            temperature=0.0,
            prefill=PLAN_PREFILL,
            constraint_schema=self.schema,
        )
        return parse_plan(gen.text, text)

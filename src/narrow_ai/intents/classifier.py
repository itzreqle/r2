"""Parse and validate model output into IntentDecision."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from narrow_ai.intents.schemas import IntentDecision, IntentName

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class IntentParseError(ValueError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if not match:
        raise IntentParseError("No JSON object found in model output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise IntentParseError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise IntentParseError("JSON root must be an object")
    return data


def parse_intent_decision(text: str) -> IntentDecision:
    data = extract_json_object(text)
    # Normalize common aliases
    if "intent" not in data and "name" in data:
        data["intent"] = data.pop("name")
    data.setdefault("parameters", {})
    data.setdefault("requires_confirmation", False)
    try:
        decision = IntentDecision.model_validate(data)
        decision.validated_params()  # validate param schema
        return decision
    except ValidationError as exc:
        raise IntentParseError(str(exc)) from exc


def heuristic_intent(user_text: str) -> IntentDecision:
    """Deterministic Phase-1 / FakeBackend routing for tests and offline use."""
    t = user_text.lower().strip()
    if any(k in t for k in ("help", "what can you", "capabilities")):
        return IntentDecision(intent=IntentName.GENERAL_HELP, parameters={})
    if "git status" in t or ("git" in t and "status" in t) or t in {"status", "git"}:
        return IntentDecision(intent=IntentName.GIT_STATUS, parameters={})
    if "find" in t or "search" in t or "files" in t:
        pattern = "*.py"
        if "*.py" in t or "python" in t:
            pattern = "*.py"
        elif "*.ts" in t or "typescript" in t:
            pattern = "*.ts"
        elif "*.md" in t or "markdown" in t:
            pattern = "*.md"
        return IntentDecision(
            intent=IntentName.SEARCH_FILES,
            parameters={"pattern": pattern, "workspace": "active"},
        )
    if t.startswith("remember ") or "remember that" in t:
        # remember KEY=VALUE or remember KEY is VALUE
        body = t.removeprefix("remember ").strip()
        if "=" in body:
            key, _, value = body.partition("=")
            return IntentDecision(
                intent=IntentName.REMEMBER_PREFERENCE,
                parameters={"key": key.strip()[:64], "value": value.strip()[:1024]},
                requires_confirmation=True,
            )
    if t.startswith("recall ") or "what is my" in t:
        key = t.removeprefix("recall ").strip() or "default"
        return IntentDecision(
            intent=IntentName.RECALL_PREFERENCE,
            parameters={"key": key[:64]},
        )
    if len(t.split()) < 2:
        return IntentDecision(
            intent=IntentName.CLARIFY_REQUEST,
            parameters={"question": "Could you rephrase that as a developer task?"},
        )
    return IntentDecision(
        intent=IntentName.UNSUPPORTED,
        parameters={"reason": "Request is outside the supported developer domain"},
    )

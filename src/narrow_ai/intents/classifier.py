"""Parse and validate model output into IntentDecision."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from narrow_ai.intents.schemas import PARAMS_BY_INTENT, IntentDecision, IntentName


class IntentParseError(ValueError):
    pass


_EXT_WORDS = {
    "python": "py",
    "py": "py",
    "typescript": "ts",
    "ts": "ts",
    "javascript": "js",
    "js": "js",
    "markdown": "md",
    "md": "md",
    "json": "json",
    "yaml": "yaml",
    "yml": "yml",
    "toml": "toml",
    "swift": "swift",
    "rust": "rs",
    "go": "go",
    "java": "java",
    "html": "html",
    "css": "css",
    "shell": "sh",
    "sh": "sh",
    "notebook": "ipynb",
    "notebooks": "ipynb",
    "ipynb": "ipynb",
    "csv": "csv",
    "txt": "txt",
    "text": "txt",
}


def normalize_decision_data(data: dict[str, Any]) -> dict[str, Any]:
    """Make model output easier to validate *without* widening what is allowed.

    - unknown parameter keys are **rejected**, not dropped: the contract is strict,
      so a model that drifts off-schema is visible (retry → heuristic) instead of
      silently "fixed"
    - bare search words become globs: "json" -> "*.json", "config" -> "*config*"
    - language words win over substring globs: "*swift*" / "swift files" -> "*.swift"
    """
    try:
        intent = IntentName(data.get("intent"))
    except ValueError:
        return data  # unknown intent → validation will reject it
    params = data.get("parameters")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise IntentParseError("parameters must be a JSON object")
    fields = PARAMS_BY_INTENT[intent].model_fields
    unknown = sorted(k for k in params if k not in fields)
    if unknown:
        raise IntentParseError(f"unknown parameters for {intent.value}: {', '.join(unknown)}")
    params = dict(params)
    if intent == IntentName.SEARCH_FILES and isinstance(params.get("pattern"), str):
        pat = params["pattern"].strip()
        # "swift", "swift files", "*swift*", ".swift" all mean files of that language.
        word = re.sub(r"\s+files?$", "", pat.lower().strip("*").lstrip(".")).strip()
        if word in _EXT_WORDS and (pat.lower().strip("*") == pat.lower() or pat.startswith("*")):
            pat = f"*.{_EXT_WORDS[word]}"
        elif pat and not any(c in pat for c in "*?["):
            pat = f"*{pat}*"
        params["pattern"] = pat
        params.setdefault("workspace", "active")
    data["parameters"] = params
    return data


_DECODER = json.JSONDecoder()


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
    # Scan for the first decodable object (non-greedy, unlike a {.*} regex).
    idx = text.find("{")
    last_err: Exception | None = None
    while idx >= 0:
        try:
            data, _ = _DECODER.raw_decode(text[idx:])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            last_err = exc
        idx = text.find("{", idx + 1)
    if last_err is not None:
        raise IntentParseError(f"Invalid JSON: {last_err}")
    raise IntentParseError("No JSON object found in model output")


def parse_intent_decision(text: str) -> IntentDecision:
    data = extract_json_object(text)
    # Normalize common aliases
    if "intent" not in data and "name" in data:
        data["intent"] = data.pop("name")
    data.setdefault("parameters", {})
    data.setdefault("requires_confirmation", False)
    data = normalize_decision_data(data)
    try:
        decision = IntentDecision.model_validate(data)
        decision.validated_params()  # validate param schema
        return decision
    except ValidationError as exc:
        raise IntentParseError(str(exc)) from exc


def heuristic_intent(user_text: str) -> IntentDecision:
    """Deterministic Phase-1 / FakeBackend routing for tests and offline use."""
    raw = user_text.strip()
    t = raw.lower()
    if any(k in t for k in ("help", "what can you", "capabilities")):
        return IntentDecision(intent=IntentName.GENERAL_HELP, parameters={})
    if "test" in t and re.search(r"\b(why|explain|what)\b", t) and re.search(
        r"\b(fail|failed|failing|broke|broken|output|error)\b", t
    ):
        return IntentDecision(intent=IntentName.EXPLAIN_TEST_OUTPUT, parameters={})
    if re.search(r"\b(run|rerun|re-run|execute)\b.*\b(tests?|test suite|specs?)\b", t):
        return IntentDecision(intent=IntentName.RUN_TESTS, parameters={})
    if "diff" in t:
        return IntentDecision(intent=IntentName.GIT_DIFF_SUMMARY, parameters={})
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
        # remember KEY=VALUE (value keeps the user's casing)
        body = raw[len("remember ") :].strip() if t.startswith("remember ") else raw
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
    if t.startswith(("how do i", "how to", "what is", "what's the difference", "why ", "explain ")):
        return IntentDecision(
            intent=IntentName.ANSWER_QUESTION, parameters={"question": raw[:1000]}
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

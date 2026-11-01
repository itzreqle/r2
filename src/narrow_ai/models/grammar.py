"""JSON schema of a complete intent decision (pure Python, no grammar engine needed).

The schema is generated from the same Pydantic parameter models that validate
decisions afterwards (``intents/schemas.py``), so decode-time constraints and
post-hoc validation cannot drift apart. It is deliberately *tighter* than the
Pydantic models where the app knows more than the model:

- only **implemented** intents are allowed (``registry.implemented()``)
- ``requires_confirmation`` is not part of the schema: the permission gate,
  not the model, decides what needs confirmation
- no extra keys anywhere (``additionalProperties: false``)

The grammar constrains *structure*, never *allowlisted values*. An earlier
version made ``open_app.app_name`` an enum of the approved apps. On the real
model, "open calculator" then decoded as ``{"app_name": "Terminal"}``: the
grammar forced a plausible but wrong action, where the honest answer was "not
an approved app". Allowlists stay in the executor, which can refuse.

Post-decode validation still runs; the grammar only removes off-schema output
before it is generated.
"""

from __future__ import annotations

import copy
from typing import Any

from narrow_ai.intents.registry import IntentRegistry
from narrow_ai.intents.schemas import PARAMS_BY_INTENT, IntentName

_DROP_KEYS = {"title", "description", "examples"}


def _clean(node: Any) -> Any:
    """Strip cosmetic keys and force closed objects (recursively)."""
    if isinstance(node, dict):
        out = {k: _clean(v) for k, v in node.items() if k not in _DROP_KEYS}
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out.setdefault("properties", {})
        return out
    if isinstance(node, list):
        return [_clean(v) for v in node]
    return node


def params_schema(intent: IntentName) -> dict:
    return _clean(copy.deepcopy(PARAMS_BY_INTENT[intent].model_json_schema()))


def decision_json_schema(registry: IntentRegistry) -> dict:
    """``anyOf`` one closed object per implemented intent: {"intent": const, "parameters": …}."""
    variants = []
    for spec in registry.implemented():
        variants.append(
            {
                "type": "object",
                "properties": {
                    "intent": {"const": spec.name.value},
                    "parameters": params_schema(spec.name),
                },
                "required": ["intent", "parameters"],
                "additionalProperties": False,
            }
        )
    return {"anyOf": variants}


def schema_allows(schema: dict, decision: dict) -> bool:
    """Tiny structural check of a decision against :func:`decision_json_schema`.

    Covers what the schema uses (const/enum/type/required/additionalProperties,
    string lengths), so tests can check examples without a JSON-schema library.
    """
    return any(_matches(v, decision) for v in schema["anyOf"])


def _matches(schema: dict, value: Any) -> bool:
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        return all(_matches(props[k], v) for k, v in value.items() if k in props)
    if t == "string":
        if not isinstance(value, str):
            return False
        if len(value) < schema.get("minLength", 0):
            return False
        return len(value) <= schema.get("maxLength", 1 << 30)
    return True

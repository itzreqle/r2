"""Load evaluation JSONL cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from narrow_ai.intents.schemas import IntentName


class EvalCase(BaseModel):
    id: str
    utterance: str
    expected_intent: IntentName
    expected_parameters: dict[str, Any] = Field(default_factory=dict)


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "id" not in data:
                data["id"] = f"case-{i}"
            cases.append(EvalCase.model_validate(data))
    return cases

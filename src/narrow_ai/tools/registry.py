"""Tool registry: one handler per intent, looked up by the executor.

A :class:`ToolSpec` only says *how* to run an already-authorized, validated
intent. It is never consulted to decide *whether* to run one. That stays in
``PermissionGate`` and ``ToolExecutor.execute`` (gate → validate → precheck →
confirm → handler), the single execution path that the multi-step agent
runtime reuses unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from narrow_ai.intents.schemas import PARAMS_BY_INTENT, IntentName
from narrow_ai.tools.result import ToolResult

if TYPE_CHECKING:
    from narrow_ai.intents.registry import IntentRegistry
    from narrow_ai.memory.preferences import PreferenceStore
    from narrow_ai.settings import Settings


@dataclass(frozen=True)
class ToolContext:
    settings: Settings
    intents: IntentRegistry
    workspace: Path
    timeout: int
    max_chars: int
    preferences: PreferenceStore | None = None
    answerer: Callable[[str], str] | None = None


Handler = Callable[[BaseModel, ToolContext], ToolResult]
# Runs after validation and *before* asking for confirmation: refuse early so the
# user is never asked to approve something that would be refused anyway.
Precheck = Callable[[BaseModel, ToolContext], ToolResult | None]


@dataclass(frozen=True)
class ToolSpec:
    intent: IntentName
    handler: Handler
    precheck: Precheck | None = None

    @property
    def params_model(self) -> type[BaseModel]:
        return PARAMS_BY_INTENT[self.intent]


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._by_intent: dict[IntentName, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.intent in self._by_intent:
            raise ValueError(f"duplicate tool for {spec.intent}")
        self._by_intent[spec.intent] = spec

    def get(self, intent: IntentName) -> ToolSpec | None:
        return self._by_intent.get(intent)

    def intents(self) -> set[IntentName]:
        return set(self._by_intent)

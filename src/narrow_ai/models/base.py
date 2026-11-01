"""Model backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    load_time_ms: float = 0.0
    generate_time_ms: float = 0.0
    token_count: int | None = None
    backend: str = ""
    meta: dict[str, Any] | None = None


class ModelBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        return None

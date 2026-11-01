"""Model backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Full decoded text (prefill included) plus performance metrics."""

    text: str
    load_time_ms: float = 0.0
    generate_time_ms: float = 0.0
    token_count: int | None = None
    backend: str = ""
    meta: dict[str, Any] | None = None
    # Performance / KV-cache metrics (0 when not applicable)
    ttft_ms: float = 0.0
    prompt_tokens: int = 0  # tokens actually prefilled this call
    cached_tokens: int = 0  # tokens served from the KV cache
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    cache_source: str = "none"

    def metrics(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ttft_ms": round(self.ttft_ms, 1),
            "total_ms": round(self.generate_time_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "generated_tokens": self.token_count or 0,
            "prompt_tps": round(self.prompt_tps, 1),
            "generation_tps": round(self.generation_tps, 1),
            "peak_memory_gb": round(self.peak_memory_gb, 3),
            "cache_source": self.cache_source,
        }


@dataclass
class CacheStats:
    mode: str = "none"
    prefix_tokens: int = 0
    source: str = "none"  # none | built | disk
    build_ms: float = 0.0
    load_ms: float = 0.0
    hits: int = 0
    misses: int = 0
    path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ModelBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prefill: str = "",
        constrain_intents: list[str] | None = None,
    ) -> GenerationResult:
        """Generate the assistant reply.

        ``prefill`` is text forced at the start of the assistant turn; the returned
        ``text`` always includes it. ``constrain_intents`` restricts the intent name
        that follows the prefill (backends may ignore it).
        """
        raise NotImplementedError

    def warmup(self, system_prompt: str) -> None:  # noqa: B027 - optional hook
        """Optionally precompute state for a system prompt (e.g. KV prefix cache)."""
        return None

    def cache_stats(self) -> CacheStats:
        return CacheStats()

    def close(self) -> None:
        return None

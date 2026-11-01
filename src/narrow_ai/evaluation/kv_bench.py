"""KV cache experiment: none vs prefix vs persistent (disk) prefix cache.

Measures, per mode:
- setup cost (prefill of the system prompt, or load from disk)
- per-request TTFT / total latency / prefilled tokens
- output equivalence against the no-cache baseline (greedy decoding)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from narrow_ai.assistant.orchestrator import IntentEngine
from narrow_ai.intents.registry import default_registry
from narrow_ai.settings import Settings


@dataclass
class ModeResult:
    mode: str
    setup_ms: float
    setup_source: str
    prefix_tokens: int
    cache_mb: float
    ttft_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "setup_ms": round(self.setup_ms, 1),
            "setup_source": self.setup_source,
            "prefix_tokens": self.prefix_tokens,
            "cache_mb": round(self.cache_mb, 1),
            "ttft_ms_mean": round(statistics.mean(self.ttft_ms), 1),
            "ttft_ms_p50": round(statistics.median(self.ttft_ms), 1),
            "total_ms_mean": round(statistics.mean(self.total_ms), 1),
            "prefilled_tokens_mean": round(statistics.mean(self.prompt_tokens), 1),
        }


def run_kv_benchmark(
    backend: Any,
    settings: Settings,
    utterances: list[str],
    modes: tuple[str, ...] = ("none", "prefix", "persistent"),
    repeats: int = 1,
) -> dict[str, Any]:
    """``backend`` must be an MlxBackend (has set_cache_mode / kv)."""
    engine = IntentEngine(settings, backend, default_registry())
    results: dict[str, ModeResult] = {}
    for mode in modes:
        backend.set_cache_mode(mode)
        if mode == "persistent":
            # Build + save once, then simulate an app restart: a fresh cache object
            # must load from disk instead of recomputing the prefix.
            backend.warmup(engine.system_prompt)
            backend.set_cache_mode(mode)
        t0 = time.perf_counter()
        backend.warmup(engine.system_prompt)
        setup_ms = (time.perf_counter() - t0) * 1000
        st = backend.cache_stats()
        res = ModeResult(
            mode=mode,
            setup_ms=setup_ms,
            setup_source=st.source,
            prefix_tokens=st.prefix_tokens,
            cache_mb=backend.kv.nbytes() / 1e6 if backend.kv else 0.0,
        )
        for _ in range(repeats):
            for u in utterances:
                decision, gen, _src = engine.decide(u)
                res.ttft_ms.append(gen.ttft_ms)
                res.total_ms.append(gen.generate_time_ms)
                res.prompt_tokens.append(gen.prompt_tokens)
                res.outputs.append(gen.text)
        results[mode] = res

    base = results.get("none")
    report: dict[str, Any] = {"modes": {m: r.summary() for m, r in results.items()}}
    if base:
        for m, r in results.items():
            same = sum(a == b for a, b in zip(base.outputs, r.outputs, strict=True))
            report["modes"][m]["output_match_vs_none"] = round(same / len(base.outputs), 3)
            if m != "none":
                b_ttft = statistics.mean(base.ttft_ms)
                r_ttft = statistics.mean(r.ttft_ms)
                report["modes"][m]["ttft_speedup_vs_none"] = round(b_ttft / r_ttft, 2)
    report["raw"] = {m: asdict(r) for m, r in results.items()}
    report["n_utterances"] = len(utterances) * repeats
    return report

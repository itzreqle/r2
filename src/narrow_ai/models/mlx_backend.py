"""MLX-LM inference backend with KV prefix caching and constrained intent decoding."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from narrow_ai.models.base import CacheStats, GenerationResult, ModelBackend
from narrow_ai.models.constrained import IntentNameConstraint
from narrow_ai.models.kv_cache import PrefixKVCache

logger = logging.getLogger(__name__)

_DECODER = json.JSONDecoder()


def _json_complete(text: str) -> bool:
    start = text.find("{")
    if start < 0:
        return False
    try:
        _DECODER.raw_decode(text[start:])
        return True
    except json.JSONDecodeError:
        return False


class MlxBackend(ModelBackend):
    name = "mlx"

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: Path | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_kv_size: int | None = None,
        cache_mode: str = "prefix",
        cache_dir: Path | None = None,
        stop_at_json: bool = True,
        load: bool = True,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_kv_size = max_kv_size
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir
        self.stop_at_json = stop_at_json
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_time_ms = 0.0
        self.kv: PrefixKVCache | None = None
        if load:
            self.load()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        try:
            import mlx_lm
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise RuntimeError("mlx-lm is not installed. Run: uv sync  (ml group)") from exc
        t0 = time.perf_counter()
        adapter = str(self.adapter_path) if self.adapter_path else None
        logger.info("Loading MLX model %s (adapter=%s)", self.model_id, adapter)
        from narrow_ai.models.hub import local_or_repo

        source = local_or_repo(self.model_id)
        if adapter:
            self._model, self._tokenizer = mlx_load(source, adapter_path=adapter)
        else:
            self._model, self._tokenizer = mlx_load(source)
        self._load_time_ms = (time.perf_counter() - t0) * 1000
        logger.info("Model loaded in %.0f ms", self._load_time_ms)
        template = getattr(self._tokenizer, "chat_template", "") or ""
        import hashlib

        self._fingerprint = {
            "model_id": self.model_id,
            "adapter": adapter or "",
            "template_sha": hashlib.sha256(template.encode()).hexdigest()[:16],
            "mlx_lm": getattr(mlx_lm, "__version__", "?"),
        }
        self.set_cache_mode(self.cache_mode)

    def set_cache_mode(self, mode: str) -> None:
        """Swap KV cache strategy at runtime (used by benchmarks / notebooks)."""
        self.cache_mode = mode
        self.kv = PrefixKVCache(
            self._model,
            mode=mode,
            cache_dir=self.cache_dir,
            base_fingerprint=self._fingerprint,
        )

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model(self) -> Any:
        return self._model

    # ------------------------------------------------------------------ prompt
    def render(self, messages: list[dict[str, str]], prefill: str = "") -> str:
        tok = self._tokenizer
        if getattr(tok, "chat_template", None) is not None:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: no <think> block, answer directly
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return prompt + prefill

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _prefix(self, messages: list[dict[str, str]], full_text: str) -> tuple[str, list[int]]:
        """System-message prefix (text, tokens) if it is a clean token prefix."""
        if not messages or messages[0].get("role") != "system":
            return "", []
        tok = self._tokenizer
        if getattr(tok, "chat_template", None) is None:
            return "", []
        prefix_text = tok.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        if not full_text.startswith(prefix_text):
            return "", []
        return prefix_text, self.encode(prefix_text)

    def warmup(self, system_prompt: str) -> None:
        if self._model is None:
            self.load()
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "hi"}]
        full = self.render(msgs)
        prefix_text, prefix_tokens = self._prefix(msgs, full)
        if prefix_tokens and self.kv is not None:
            self.kv.prepare(prefix_text, prefix_tokens)
        # Compile Metal kernels now so the user's first request is fast.
        self.generate(msgs, max_tokens=1)
        if self.kv is not None:
            self.kv.stats.hits = self.kv.stats.misses = 0

    def cache_stats(self) -> CacheStats:
        if self.kv is None:
            return CacheStats()
        self.kv.stats.extra["nbytes"] = self.kv.nbytes()
        return self.kv.stats

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prefill: str = "",
        constrain_intents: list[str] | None = None,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            self.load()
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        assert self.kv is not None
        full_text = self.render(messages, prefill)
        full_tokens = self.encode(full_text)

        prefix_text, prefix_tokens = self._prefix(messages, full_text)
        if prefix_tokens and full_tokens[: len(prefix_tokens)] == prefix_tokens:
            self.kv.prepare(prefix_text, prefix_tokens)
        cache, suffix, n_cached = self.kv.acquire(full_tokens)

        processors = None
        constraint = None
        if constrain_intents and prefill:
            constraint = IntentNameConstraint(self._tokenizer, constrain_intents, len(suffix))
            processors = [constraint]

        mt = max_tokens if max_tokens is not None else self.max_tokens
        temp = self.temperature if temperature is None else temperature
        sampler = make_sampler(temp=temp)
        kwargs: dict[str, Any] = {
            "max_tokens": mt,
            "sampler": sampler,
            "logits_processors": processors,
        }
        if cache is not None:
            kwargs["prompt_cache"] = cache
        elif self.max_kv_size:
            kwargs["max_kv_size"] = self.max_kv_size

        mx.reset_peak_memory()
        t0 = time.perf_counter()
        ttft = 0.0
        text = ""
        last = None
        try:
            for resp in stream_generate(self._model, self._tokenizer, prompt=suffix, **kwargs):
                if last is None:
                    ttft = (time.perf_counter() - t0) * 1000
                last = resp
                text += resp.text
                if self.stop_at_json and "}" in resp.text and _json_complete(prefill + text):
                    break
        finally:
            self.kv.release(cache)
        elapsed = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            text=prefill + text,
            load_time_ms=self._load_time_ms,
            generate_time_ms=elapsed,
            token_count=last.generation_tokens if last else 0,
            backend=self.name,
            meta={
                "model_id": self.model_id,
                "temperature": temp,
                "constrained_steps": constraint.masked_steps if constraint else 0,
            },
            ttft_ms=ttft,
            prompt_tokens=last.prompt_tokens if last else len(suffix),
            cached_tokens=n_cached,
            prompt_tps=last.prompt_tps if last else 0.0,
            generation_tps=last.generation_tps if last else 0.0,
            peak_memory_gb=float(mx.get_peak_memory()) / 1e9,
            cache_source=self.kv.stats.source if cache is not None else "none",
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self.kv = None

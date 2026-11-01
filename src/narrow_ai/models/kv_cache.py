"""KV cache management for R2.

Three modes (``model.cache_mode``):

- ``none``        every request prefills the full prompt (baseline).
- ``prefix``      the system prompt (instructions + few-shot examples) is prefilled
                  once into a KV cache; each request only prefills the new suffix
                  (user turn) and the cache is trimmed back afterwards.
- ``persistent``  like ``prefix`` but the prefix cache is also saved to disk
                  (safetensors) and re-loaded on the next start, so even the first
                  request after a restart skips the system-prompt prefill.

A disk cache is only valid for the exact model, adapter, chat template, prefix
text and mlx-lm version it was built with; a fingerprint in the file's metadata
is checked on load and stale caches are rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from narrow_ai.models.base import CacheStats

logger = logging.getLogger(__name__)

CACHE_MODES = ("none", "prefix", "persistent")


def fingerprint(**parts: str) -> dict[str, str]:
    fp = {k: str(v) for k, v in sorted(parts.items())}
    fp["fingerprint"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:20]
    return fp


class PrefixKVCache:
    def __init__(
        self,
        model: Any,
        mode: str = "prefix",
        cache_dir: Path | None = None,
        base_fingerprint: dict[str, str] | None = None,
    ) -> None:
        if mode not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")
        self.model = model
        self.mode = mode
        self.cache_dir = cache_dir
        self.base_fingerprint = base_fingerprint or {}
        self._cache: list[Any] | None = None
        self._prefix_tokens: list[int] = []
        self._prefix_text: str = ""
        self.stats = CacheStats(mode=mode)

    # ------------------------------------------------------------------ build
    def prepare(self, prefix_text: str, prefix_tokens: list[int]) -> None:
        """Make sure a cache for exactly this prefix is ready (build or load)."""
        if self.mode == "none":
            return
        if self._cache is not None and prefix_text == self._prefix_text:
            return
        self._cache = None
        self._prefix_text = prefix_text
        self._prefix_tokens = list(prefix_tokens)
        self.stats.prefix_tokens = len(prefix_tokens)
        fp = fingerprint(
            **self.base_fingerprint,
            prefix_sha=hashlib.sha256(prefix_text.encode()).hexdigest(),
            prefix_tokens=str(len(prefix_tokens)),
        )
        if self.mode == "persistent" and self._try_load(fp):
            return
        self._build()
        if self.mode == "persistent":
            self._save(fp)

    def _build(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        t0 = time.perf_counter()
        cache = make_prompt_cache(self.model)
        tokens = mx.array(self._prefix_tokens)
        step = 2048
        for i in range(0, len(self._prefix_tokens), step):
            self.model(tokens[i : i + step][None], cache=cache)
            mx.eval([c.state for c in cache])
        self._cache = cache
        self.stats.build_ms = (time.perf_counter() - t0) * 1000
        self.stats.source = "built"
        logger.info(
            "KV prefix cache built: %d tokens in %.0f ms",
            len(self._prefix_tokens),
            self.stats.build_ms,
        )

    def _path(self, fp: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"prefix-{fp['fingerprint']}.safetensors"

    def _try_load(self, fp: dict[str, str]) -> bool:
        from mlx_lm.models.cache import load_prompt_cache

        path = self._path(fp)
        if path is None or not path.is_file():
            return False
        t0 = time.perf_counter()
        try:
            cache, meta = load_prompt_cache(str(path), return_metadata=True)
        except Exception as exc:  # noqa: BLE001 - corrupt cache → rebuild
            logger.warning("could not load KV cache %s: %s", path, exc)
            return False
        import mlx.core as mx

        mx.eval([c.state for c in cache])  # materialize (mx.load is lazy) for honest timing
        if meta.get("fingerprint") != fp["fingerprint"]:
            logger.warning("KV cache fingerprint mismatch, rebuilding")
            return False
        if cache[0].offset != len(self._prefix_tokens):
            logger.warning("KV cache length mismatch, rebuilding")
            return False
        self._cache = cache
        self.stats.load_ms = (time.perf_counter() - t0) * 1000
        self.stats.source = "disk"
        self.stats.path = str(path)
        logger.info("KV prefix cache loaded from disk in %.0f ms", self.stats.load_ms)
        return True

    def _save(self, fp: dict[str, str]) -> None:
        from mlx_lm.models.cache import save_prompt_cache

        path = self._path(fp)
        if path is None or self._cache is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        save_prompt_cache(str(path), self._cache, metadata=fp)
        self.stats.path = str(path)
        logger.info("KV prefix cache saved to %s", path)

    # ------------------------------------------------------------------ use
    def acquire(self, full_tokens: list[int]) -> tuple[list[Any] | None, list[int], int]:
        """Return (cache, tokens_still_to_prefill, n_cached_tokens)."""
        n = len(self._prefix_tokens)
        if (
            self.mode == "none"
            or self._cache is None
            or len(full_tokens) <= n
            or full_tokens[:n] != self._prefix_tokens
        ):
            if self.mode != "none":
                self.stats.misses += 1
            return None, full_tokens, 0
        self.stats.hits += 1
        return self._cache, full_tokens[n:], n

    def release(self, cache: list[Any] | None) -> None:
        """Trim the cache back to the shared prefix so it can be reused."""
        if cache is None:
            return
        from mlx_lm.models.cache import trim_prompt_cache

        extra = cache[0].offset - len(self._prefix_tokens)
        if extra > 0:
            trim_prompt_cache(cache, extra)

    def nbytes(self) -> int:
        if self._cache is None:
            return 0
        return int(sum(c.nbytes for c in self._cache if hasattr(c, "nbytes")))

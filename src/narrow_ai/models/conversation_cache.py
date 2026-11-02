"""Conversation-level KV reuse: token-prefix matching across turns.

The prefix cache (``kv_cache.PrefixKVCache``) only covers the system prompt, so every
request still prefills its history. Here the KV state of each intent-routing prompt is
kept after the call, and the next prompt reuses the longest token prefix it shares with a
stored one. Storage and matching are mlx-lm's ``LRUPromptCache``, a token trie with LRU
eviction that hands out copies.

What makes reuse safe:

* Matching is on exact token ids, so a reused position is the KV that prefilling the
  same tokens would produce. Any token that differs, along with everything after it,
  is prefilled again.
* Stored caches cover the *prompt only*. Generated tokens are trimmed off before
  storing, because the next prompt renders that reply differently: Qwen3 drops the
  empty think block from earlier assistant turns.
* Entries are copies (MLX arrays are copy-on-write), so the shared system-prefix cache
  is never mutated.
* Non-trimmable caches (for example rotating windows) are never stored.

When reuse helps: history grows by one (user, decision) pair per turn. A sliding window
shifts history once it is full, so only the system prefix then matches. See
``ModelSettings.history_window`` and ``docs/EVALUATION.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_KEY = "r2"  # one model per backend; the trie is cleared whenever the model changes


@dataclass
class ConversationStats:
    hits: int = 0
    misses: int = 0
    reused_tokens: int = 0  # beyond the system prefix, summed over hits
    stored: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "conversation_hits": self.hits,
            "conversation_misses": self.misses,
            "conversation_reused_tokens": self.reused_tokens,
            "conversation_stored": self.stored,
        }


def _offset(cache: list[Any]) -> int:
    return int(getattr(cache[0], "offset", 0)) if cache else 0


def _trimmable(cache: list[Any]) -> bool:
    return bool(cache) and all(
        getattr(c, "is_trimmable", lambda: False)() for c in cache
    )


def _trim(cache: list[Any], n: int) -> None:
    if n > 0:
        for c in cache:
            c.trim(n)


class ConversationCache:
    """Wraps ``LRUPromptCache`` with R2's rules: prompt-only entries, at least one
    token left to prefill, and nothing that can't be trimmed."""

    def __init__(self, max_entries: int = 1, max_bytes: int = 1 << 30) -> None:
        from mlx_lm.models.cache import LRUPromptCache

        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lru = LRUPromptCache(max_size=max_entries, max_bytes=max_bytes)
        self.stats = ConversationStats()

    def __len__(self) -> int:
        return len(self._lru)

    @property
    def nbytes(self) -> int:
        return int(self._lru.nbytes)

    def clear(self) -> None:
        self._lru.trim_to(n_sequences=0)

    def fetch(self, tokens: list[int], *, min_reuse: int = 0) -> tuple[list[Any] | None, list[int]]:
        """(cache copy, tokens still to prefill), or (None, tokens) if no stored prompt
        shares more than ``min_reuse`` tokens with ``tokens``."""
        if not tokens:
            return None, tokens
        cache, rest = self._lru.fetch_nearest_cache(_KEY, list(tokens))
        if cache is None:
            self.stats.misses += 1
            return None, tokens
        if not rest:  # identical prompt: re-feed the last token so there is an input
            if not _trimmable(cache):
                self.stats.misses += 1
                return None, tokens
            _trim(cache, 1)
            rest = list(tokens[-1:])
        reused = len(tokens) - len(rest)
        if reused <= min_reuse or _offset(cache) != reused:
            self.stats.misses += 1
            return None, tokens
        self.stats.hits += 1
        self.stats.reused_tokens += reused - min_reuse
        return cache, list(rest)

    def store(self, tokens: list[int], cache: list[Any]) -> bool:
        """Keep ``cache`` (which must hold at least ``tokens``) for later prompts,
        trimmed back to exactly ``tokens``. The caller must not use it afterwards."""
        if not tokens or not _trimmable(cache):
            return False
        extra = _offset(cache) - len(tokens)
        if extra < 0:  # generation was cut short before the prompt was processed
            return False
        _trim(cache, extra)
        self._lru.insert_cache(_KEY, list(tokens), cache)
        self.stats.stored += 1
        return True

"""Conversation-level KV reuse, with fake cache layers (no model, no Metal work)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx_lm.models.cache")

from narrow_ai.memory.store import ConversationStore  # noqa: E402
from narrow_ai.models.conversation_cache import ConversationCache  # noqa: E402


class FakeLayer:
    """Holds the token ids it has 'seen', like a KV layer holds their keys."""

    def __init__(self, tokens=None, trimmable=True):
        self.seen = list(tokens or [])
        self._trimmable = trimmable

    @property
    def offset(self):
        return len(self.seen)

    @property
    def nbytes(self):
        return 8 * len(self.seen)

    def is_trimmable(self):
        return self._trimmable

    def trim(self, n):
        n = min(n, len(self.seen))
        del self.seen[len(self.seen) - n:]
        return n

    def feed(self, tokens):
        self.seen += list(tokens)


def run(tokens, generated=(), layers=2, trimmable=True):
    """A cache after prefilling ``tokens`` and decoding ``generated``."""
    return [FakeLayer(list(tokens) + list(generated), trimmable) for _ in range(layers)]


SYSTEM = list(range(100))


def test_miss_when_empty():
    cc = ConversationCache()
    cache, rest = cc.fetch(SYSTEM + [1, 2])
    assert cache is None and rest == SYSTEM + [1, 2]
    assert cc.stats.misses == 1


def test_next_turn_reuses_previous_prompt_not_generated_tokens():
    cc = ConversationCache()
    turn1 = SYSTEM + [500, 501, 502]
    assert cc.store(turn1, run(turn1, generated=[900, 901, 902]))
    # next prompt = turn-1 prompt + its (re-rendered) reply + the new user turn
    turn2 = turn1 + [700, 701, 600, 601]
    cache, rest = cc.fetch(turn2, min_reuse=len(SYSTEM))
    assert cache is not None
    assert rest == [700, 701, 600, 601]
    assert all(layer.seen == turn1 for layer in cache)  # generated tokens were trimmed off
    assert cc.stats.hits == 1 and cc.stats.reused_tokens == 3


def test_divergence_reuses_only_the_common_prefix():
    cc = ConversationCache()
    old = SYSTEM + [1, 2, 3, 4]
    cc.store(old, run(old))
    new = SYSTEM + [1, 2, 9, 9]
    cache, rest = cc.fetch(new)
    assert cache is not None and rest == [9, 9]
    assert all(layer.seen == SYSTEM + [1, 2] for layer in cache)


def test_no_gain_over_system_prefix_is_a_miss():
    cc = ConversationCache()
    old = SYSTEM + [1, 2, 3]
    cc.store(old, run(old))
    cache, rest = cc.fetch(SYSTEM + [7, 8], min_reuse=len(SYSTEM))
    assert cache is None and rest == SYSTEM + [7, 8]


def test_identical_prompt_leaves_one_token_to_prefill():
    cc = ConversationCache()
    p = SYSTEM + [1, 2, 3]
    cc.store(p, run(p))
    cache, rest = cc.fetch(p)
    assert rest == [3]
    assert all(layer.offset == len(p) - 1 for layer in cache)


def test_fetch_returns_a_copy():
    cc = ConversationCache()
    p = SYSTEM + [1, 2]
    stored = run(p)
    cc.store(p, stored)
    cache, _ = cc.fetch(p + [3])
    cache[0].feed([42, 43])  # decoding into the copy...
    again, rest = cc.fetch(p + [3])
    assert again[0].seen == p and rest == [3]  # ...doesn't change what is stored


def test_untrimmable_and_short_caches_are_not_stored():
    cc = ConversationCache()
    p = SYSTEM + [1]
    assert not cc.store(p, run(p, trimmable=False))  # e.g. a rotating window
    assert not cc.store(p, run(SYSTEM))  # generation stopped before the prompt was in
    assert len(cc) == 0


def test_lru_bound_and_clear():
    cc = ConversationCache(max_entries=2)
    for k in range(4):
        p = SYSTEM + [1000 + k]
        cc.store(p, run(p))
    assert len(cc) == 2
    cc.clear()
    assert len(cc) == 0


@pytest.mark.parametrize(
    ("window", "sizes"),
    [("sliding", [0, 1, 2, 3, 3, 3, 3]), ("block", [0, 1, 2, 3, 1, 2, 3])],
)
def test_history_window(window, sizes):
    store = ConversationStore(max_turns=3, window=window)
    seen = []
    for i in range(7):
        seen.append(len(store.messages) // 2)
        store.add("user", f"u{i}")
        store.add("assistant", f"a{i}")
    assert seen == sizes
    if window == "block":
        assert [m.content for m in store.messages] == ["u6", "a6"]


def test_zero_history_turns_keeps_no_history():
    store = ConversationStore(max_turns=0)
    for i in range(3):
        store.add("user", f"u{i}")
        store.add("assistant", f"a{i}")
    assert store.as_chat() == []

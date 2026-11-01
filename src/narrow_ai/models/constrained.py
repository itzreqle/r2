"""Constrained decoding for the intent name.

The assistant turn is prefilled with ``{"intent": "`` so the very next tokens must
spell an intent. This logits processor masks every token that would make the
generated text diverge from the allowed names, so the model *cannot* produce an
unknown intent. Once a full name plus its closing quote is produced, the
processor switches off and parameters are generated freely (then validated by
Pydantic).

Works with ``mlx_lm`` logits processors: ``processor(tokens, logits) -> logits``
where ``tokens`` = prompt tokens fed this call + generated tokens.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_VOCAB_CACHE: dict[int, list[str]] = {}
_CANDIDATE_CACHE: dict[tuple[int, tuple[str, ...]], tuple[list, dict]] = {}


def vocab_strings(tokenizer: Any) -> list[str]:
    """Decoded string of every token id (cached per tokenizer)."""
    key = id(tokenizer)
    if key not in _VOCAB_CACHE:
        size = (
            len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else tokenizer.vocab_size
        )
        ids = [[i] for i in range(size)]
        _VOCAB_CACHE[key] = list(tokenizer.batch_decode(ids))
    return _VOCAB_CACHE[key]


class IntentNameConstraint:
    """Allow only token sequences that spell one of ``names`` followed by ``"``."""

    def __init__(self, tokenizer: Any, names: list[str], n_prompt: int) -> None:
        self.tokenizer = tokenizer
        self.targets = [n + '"' for n in names]
        self.n_prompt = n_prompt
        self.done = False
        self.masked_steps = 0
        key = (id(tokenizer), tuple(self.targets))
        if key not in _CANDIDATE_CACHE:
            allowed_chars = set("".join(self.targets)) | set(", }")
            vocab = vocab_strings(tokenizer)
            # Pre-filter once: only tokens made of characters that can appear here.
            cands = [(tid, s) for tid, s in enumerate(vocab) if s and set(s) <= allowed_chars]
            _CANDIDATE_CACHE[key] = (cands, {})
        # allowed-id lists are shared across calls (same names → same answers)
        self._candidates, self._allowed_cache = _CANDIDATE_CACHE[key]

    def allowed_ids(self, prefix: str) -> list[int]:
        if prefix not in self._allowed_cache:
            out = []
            for tid, s in self._candidates:
                ps = prefix + s
                if any(t.startswith(ps) or ps.startswith(t) for t in self.targets):
                    out.append(tid)
            self._allowed_cache[prefix] = out
        return self._allowed_cache[prefix]

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if self.done:
            return logits
        import mlx.core as mx

        generated = tokens[self.n_prompt :].tolist() if tokens.size > self.n_prompt else []
        text = self.tokenizer.decode(generated) if generated else ""
        if any(text.startswith(t) for t in self.targets):
            self.done = True
            return logits
        allowed = self.allowed_ids(text)
        if not allowed:  # cannot help any more; validation will catch problems
            self.done = True
            return logits
        vocab_size = logits.shape[-1]
        mask = np.full((vocab_size,), -np.inf, dtype=np.float32)
        mask[[a for a in allowed if a < vocab_size]] = 0.0
        self.masked_steps += 1
        return logits + mx.array(mask)

"""Full JSON-schema constrained decoding for MLX via llguidance.

Where :mod:`narrow_ai.models.constrained` only constrains the intent *name*,
this logits processor constrains the whole decision object (intent, parameter
names, types, enums, lengths, no extra keys) to the schema from
:mod:`narrow_ai.models.grammar`.

How it works per decode step: the llguidance matcher consumes the tokens sampled
so far, fills a packed bitmask of the tokens that keep the output inside the
grammar (~tens of µs on CPU), and llguidance's Metal kernel sets every other
logit to -inf.

**Fail closed.** If the matcher errors, or no token is allowed, only EOS stays
possible. The (incomplete) output then fails to parse and goes through the usual
retry → heuristic path; it is never "unmasked" to decode freely.

The assistant turn is still prefilled with ``{"intent": "``. The matcher consumes
that prefill first. llguidance matches *bytes*, so any tokenization of the
prefill is accepted, and the forced tokens stay in the cheap prefill pass instead
of costing one decode step each.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

_LLG_TOKENIZERS: dict[int, Any] = {}
_MATCHERS: dict[tuple[int, str], Any] = {}

GRAMMAR_OPTIONS = {
    # Match the few-shot examples, which are json.dumps() output.
    "item_separator": ", ",
    "key_separator": ": ",
    "whitespace_flexible": False,
}


class GrammarUnavailable(RuntimeError):
    pass


def require_llguidance() -> Any:
    try:
        import llguidance  # noqa: F401
        import llguidance.hf  # noqa: F401
        import llguidance.numpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on install
        raise GrammarUnavailable(
            "model.constraint = 'json' needs llguidance: uv sync --group grammar"
        ) from exc
    return llguidance


def eos_ids(tokenizer: Any) -> list[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return sorted(int(i) for i in ids)
    eid = getattr(tokenizer, "eos_token_id", None)
    return [int(eid)] if eid is not None else []


def eos_only(logits: Any, ids: list[int]) -> Any:
    """Mask everything except EOS (the fail-closed outcome)."""
    import mlx.core as mx

    vocab = logits.shape[-1]
    mask = np.full((vocab,), -np.inf, dtype=np.float32)
    mask[[i for i in ids if i < vocab]] = 0.0
    return logits + mx.array(mask)


def llg_tokenizer(tokenizer: Any) -> Any:
    """llguidance view of the HF tokenizer (walks the vocab once; cached)."""
    key = id(tokenizer)
    if key not in _LLG_TOKENIZERS:
        llg = require_llguidance()
        hf = getattr(tokenizer, "_tokenizer", tokenizer)  # unwrap mlx-lm TokenizerWrapper
        _LLG_TOKENIZERS[key] = llg.hf.from_tokenizer(hf, eos_token=eos_ids(tokenizer) or None)
    return _LLG_TOKENIZERS[key]


def compile_grammar(schema: dict) -> str:
    llg = require_llguidance()
    return llg.LLMatcher.grammar_from_json_schema(schema, defaults=GRAMMAR_OPTIONS)


def _pristine_matcher(tokenizer: Any, schema: dict) -> Any:
    """One compiled matcher per (tokenizer, schema); requests use deep copies."""
    key = (id(tokenizer), json.dumps(schema, sort_keys=True))
    if key not in _MATCHERS:
        llg = require_llguidance()
        matcher = llg.LLMatcher(llg_tokenizer(tokenizer), compile_grammar(schema), log_level=0)
        if matcher.is_error():
            raise ValueError(f"invalid decision grammar: {matcher.get_error()}")
        _MATCHERS[key] = matcher
    return _MATCHERS[key]


class JsonSchemaConstraint:
    """mlx-lm logits processor: ``processor(tokens, logits) -> logits``.

    ``tokens`` is the prompt fed this call (``n_prompt`` tokens) followed by the
    tokens generated so far.
    """

    def __init__(self, tokenizer: Any, schema: dict, n_prompt: int, prefill: str = "") -> None:
        llg = require_llguidance()
        self._apply = None  # resolved lazily (imports mlx)
        self.tokenizer = tokenizer
        self.n_prompt = n_prompt
        self.eos = eos_ids(tokenizer)
        self.matcher = _pristine_matcher(tokenizer, schema).deep_copy()
        ltok = llg_tokenizer(tokenizer)
        self.bitmask = llg.numpy.allocate_token_bitmask(1, ltok.vocab_size)
        self._fill = llg.numpy.fill_next_token_bitmask
        self.consumed = 0
        self.masked_steps = 0
        self.failed = ""
        if prefill and not self.matcher.consume_tokens(ltok.tokenize_str(prefill)):
            self.failed = f"prefill rejected by grammar: {self.matcher.get_error()}"

    def _fail(self, why: str, logits: Any) -> Any:
        if not self.failed:
            self.failed = why
        return eos_only(logits, self.eos)

    def __call__(self, tokens: Any, logits: Any) -> Any:
        if self.failed:
            return eos_only(logits, self.eos)
        generated = tokens[self.n_prompt :].tolist() if tokens.size > self.n_prompt else []
        for tok in generated[self.consumed :]:
            if not self.matcher.consume_token(int(tok)):
                return self._fail(f"token {tok} rejected: {self.matcher.get_error()}", logits)
        self.consumed = len(generated)
        if self.matcher.is_error():
            return self._fail(self.matcher.get_error(), logits)
        if self.matcher.is_stopped():
            return eos_only(logits, self.eos)
        self._fill(self.matcher, self.bitmask, 0)
        if not self.bitmask.any():
            return self._fail("grammar allows no token", logits)
        if self._apply is None:
            from llguidance.mlx import apply_token_bitmask

            self._apply = apply_token_bitmask
        out = self._apply(logits, self.bitmask)
        self.masked_steps += 1
        return out.reshape(logits.shape)

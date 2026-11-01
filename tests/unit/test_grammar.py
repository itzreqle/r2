"""Decision JSON schema + llguidance constraint (tiny in-memory tokenizer, no model)."""

from __future__ import annotations

import json

import pytest

from narrow_ai.intents.registry import default_registry
from narrow_ai.intents.schemas import IntentName
from narrow_ai.models.grammar import decision_json_schema, params_schema, schema_allows
from narrow_ai.models.prompt_templates import FEW_SHOT, INTENT_PREFILL


@pytest.fixture(scope="module")
def schema() -> dict:
    return decision_json_schema(default_registry())


# --------------------------------------------------------------------- schema (pure)
def test_schema_covers_exactly_the_implemented_intents(schema):
    consts = {v["properties"]["intent"]["const"] for v in schema["anyOf"]}
    implemented = {s.name.value for s in default_registry().implemented()}
    assert consts == implemented
    assert "run_tests" not in consts and "explain_test_output" not in consts


def test_every_object_is_closed_and_has_no_confirmation_flag(schema):
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            assert "requires_confirmation" not in node.get("properties", {})
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)


def test_all_few_shot_examples_satisfy_the_schema(schema):
    for utt, dec in FEW_SHOT:
        assert schema_allows(schema, dec), utt


@pytest.mark.parametrize(
    "decision",
    [
        {"intent": "git_status", "parameters": {"verbose": True}},
        {"intent": "read_file", "parameters": {}},
        {"intent": "read_file", "parameters": {"path": ""}},
        {"intent": "open_app", "parameters": {}},
        {"intent": "run_tests", "parameters": {"command_id": "unit"}},
        {"intent": "git_status", "parameters": {}, "requires_confirmation": False},
        {"intent": "hack", "parameters": {}},
    ],
)
def test_schema_rejects_off_contract_decisions(schema, decision):
    assert not schema_allows(schema, decision)


def test_allowlisted_values_are_not_forced_by_the_grammar(schema):
    """Regression: an app enum made "open calculator" decode as Terminal on the real model.
    The grammar must let the model say "Calculator" so the executor can refuse it."""
    s = params_schema(IntentName.OPEN_APP)
    assert "enum" not in s["properties"]["app_name"]
    assert schema_allows(schema, {"intent": "open_app", "parameters": {"app_name": "Calculator"}})


# --------------------------------------------------------------------- llguidance matcher
llguidance = pytest.importorskip("llguidance")
mx = pytest.importorskip("mlx.core")


class TinyTokenizer:
    """Byte-level BPE trained in memory, wrapped like mlx-lm's TokenizerWrapper."""

    def __init__(self) -> None:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        corpus = [json.dumps(d) for _, d in FEW_SHOT] * 20 + [INTENT_PREFILL] * 20
        trainer = trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(corpus, trainer)
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
        self.eos_token_ids = {self._tokenizer.eos_token_id}
        self.vocab_size = len(self._tokenizer.get_vocab())

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def __getattr__(self, name: str):  # like TokenizerWrapper: delegate to the HF tokenizer
        return getattr(self._tokenizer, name)


@pytest.fixture(scope="module")
def tiny() -> TinyTokenizer:
    return TinyTokenizer()


def _drive(tiny: TinyTokenizer, schema: dict, completion: str):
    """Feed `completion` token by token; return (constraint, list of 'was it allowed')."""
    from narrow_ai.models.json_constraint import JsonSchemaConstraint

    prompt = [1, 2, 3]
    c = JsonSchemaConstraint(tiny, schema, n_prompt=len(prompt), prefill=INTENT_PREFILL)
    allowed: list[bool] = []
    generated: list[int] = []
    for tok in tiny.encode(completion):
        logits = mx.zeros((1, tiny.vocab_size + 7))  # model vocab is padded
        out = c(mx.array(prompt + generated), logits)
        allowed.append(bool(out[0, tok].item() > -1e30))
        generated.append(tok)
    final = c(mx.array(prompt + generated), mx.zeros((1, tiny.vocab_size + 7)))
    return c, allowed, final


def test_valid_decision_is_allowed_every_step(tiny, schema):
    c, allowed, final = _drive(tiny, schema, 'search_files", "parameters": {"pattern": "*.py"}}')
    assert all(allowed) and not c.failed and c.masked_steps > 0
    eos = next(iter(tiny.eos_token_ids))
    finite = [i for i in range(final.shape[-1]) if final[0, i].item() > -1e30]
    assert finite == [eos]  # complete object: only EOS may follow
    assert final[0, tiny.vocab_size + 3].item() < -1e30  # padded logits stay masked


@pytest.mark.parametrize(
    "completion",
    [
        'git_status", "parameters": {"verbose": true}}',  # extra key
        'open_app", "parameters": {"app": "Cursor"}}',  # wrong key
        'run_tests", "parameters": {"command_id": "x"}}',  # unimplemented
        'git_status", "parameters": {}, "requires_confirmation": false}',
    ],
)
def test_off_schema_tokens_are_masked(tiny, schema, completion):
    c, allowed, _ = _drive(tiny, schema, completion)
    assert not all(allowed)


def test_rejected_token_fails_closed_to_eos(tiny, schema):
    """If an off-grammar token is ever sampled, only EOS remains (never unmasked)."""
    c, _, final = _drive(tiny, schema, 'hack_the_planet"}')
    assert c.failed
    eos = next(iter(tiny.eos_token_ids))
    finite = [i for i in range(final.shape[-1]) if final[0, i].item() > -1e30]
    assert finite == [eos]


def test_intent_constraint_fails_closed_when_nothing_allowed(tiny):
    from narrow_ai.models.constrained import IntentNameConstraint

    c = IntentNameConstraint(tiny, ["git_status"], n_prompt=1)
    weird = tiny.encode("zzz")  # spells nothing that can continue an intent name
    out = c(mx.array([0] + weird), mx.zeros((1, tiny.vocab_size)))
    eos = next(iter(tiny.eos_token_ids))
    finite = [i for i in range(out.shape[-1]) if out[0, i].item() > -1e30]
    assert c.failed and finite == [eos]

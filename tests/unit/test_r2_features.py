"""R2 sprint features: parsing/normalization, prompts, executor tools, constraint."""

from __future__ import annotations

from pathlib import Path

import pytest

from narrow_ai.intents.classifier import (
    IntentParseError,
    extract_json_object,
    heuristic_intent,
    parse_intent_decision,
)
from narrow_ai.intents.registry import default_registry
from narrow_ai.intents.schemas import IntentDecision, IntentName
from narrow_ai.memory.preferences import PreferenceStore
from narrow_ai.models.prompt_templates import ANSWER_SYSTEM, build_system_prompt
from narrow_ai.settings import Settings
from narrow_ai.tools.executor import ToolExecutor, normalize_key


# ---------------------------------------------------------------- parsing
def test_first_json_object_not_greedy():
    text = 'noise {"intent": "git_status", "parameters": {}} trailing {"x": 1}'
    assert extract_json_object(text)["intent"] == "git_status"


@pytest.mark.parametrize(
    "text",
    [
        '{"intent": "git_diff_summary", "parameters": {"lines": "1"}}',
        '{"intent": "read_file", "parameters": {"path": "a.py", "shell": "cat /etc/passwd"}}',
        '{"intent": "search_files", "parameters": {"pattern": "*.py", "root": "/"}}',
        '{"intent": "git_status", "parameters": ["x"]}',
    ],
)
def test_unknown_or_malformed_params_are_rejected(text):
    """Strict contract: off-schema fields are an error, never silently dropped."""
    with pytest.raises(IntentParseError):
        parse_intent_decision(text)


def test_declared_optional_params_are_accepted():
    d = parse_intent_decision(
        '{"intent": "search_files", "parameters": {"pattern": "*.py", "workspace": "active"}}'
    )
    assert d.parameters == {"pattern": "*.py", "workspace": "active"}
    assert parse_intent_decision('{"intent": "git_status"}').parameters == {}


def test_top_level_extra_fields_still_rejected():
    with pytest.raises(ValueError):
        parse_intent_decision('{"intent": "git_status", "parameters": {}, "shell": "rm -rf /"}')


@pytest.mark.parametrize(
    "raw,expected",
    [("json", "*.json"), ("python", "*.py"), ("config", "*config*"), ("*.md", "*.md")],
)
def test_search_pattern_normalization(raw, expected):
    d = parse_intent_decision(f'{{"intent": "search_files", "parameters": {{"pattern": "{raw}"}}}}')
    assert d.parameters["pattern"] == expected


def test_heuristic_keeps_value_case_and_answers_questions():
    d = heuristic_intent("Remember preferred_editor=Cursor")
    assert d.parameters["value"] == "Cursor"
    assert heuristic_intent("how do I undo a commit").intent == IntentName.ANSWER_QUESTION


# ---------------------------------------------------------------- prompts
def test_system_prompt_lists_only_implemented_intents():
    prompt = build_system_prompt("", default_registry(), approved_apps=["Xcode"])
    assert "answer_question" in prompt and "git_diff_summary" in prompt
    assert "run_tests" not in prompt  # not implemented → not offered
    assert "Approved apps: Xcode." in prompt
    assert "never claim" in ANSWER_SYSTEM


# ---------------------------------------------------------------- executor
def _executor(tmp_path: Path, **kw) -> ToolExecutor:
    s = Settings()
    s.workspace.path = str(tmp_path)
    s.tools.allow_read_file = True
    s.tools.allow_open_app = True
    s.tools.approved_apps = ["Xcode"]
    return ToolExecutor(s, default_registry(), PreferenceStore(tmp_path / "p.db"), **kw)


def test_answer_question_uses_answerer_and_never_executes(tmp_path):
    ex = _executor(tmp_path, answerer=lambda q: f"Answer to: {q}")
    r = ex.execute(
        IntentDecision(intent=IntentName.ANSWER_QUESTION, parameters={"question": "why?"})
    )
    assert r.ok and r.message == "Answer to: why?" and r.speech == r.message


def test_read_file_refuses_secrets_and_escape(tmp_path):
    (tmp_path / ".env.local").write_text("TOKEN=x")
    ex = _executor(tmp_path)
    secret = ex.execute(
        IntentDecision(intent=IntentName.READ_FILE, parameters={"path": ".env.local"})
    )
    escape = ex.execute(
        IntentDecision(intent=IntentName.READ_FILE, parameters={"path": "../../etc/passwd"})
    )
    assert not secret.ok and not escape.ok


def test_read_file_ok(tmp_path):
    (tmp_path / "README.md").write_text("hello\nworld\n")
    r = _executor(tmp_path).execute(
        IntentDecision(intent=IntentName.READ_FILE, parameters={"path": "README.md"})
    )
    assert r.ok and "hello" in r.message


def test_open_app_requires_confirmation_and_allowlist(tmp_path):
    ex = _executor(tmp_path)
    first = ex.execute(IntentDecision(intent=IntentName.OPEN_APP, parameters={"app_name": "xcode"}))
    assert first.needs_confirmation and not first.ok
    other = _executor(tmp_path, confirmed=True).execute(
        IntentDecision(intent=IntentName.OPEN_APP, parameters={"app_name": "Mail"})
    )
    assert not other.ok and "approved" in other.message


def test_recall_is_fuzzy_and_keys_normalized(tmp_path):
    ex = _executor(tmp_path, confirmed=True)
    ex.execute(
        IntentDecision(
            intent=IntentName.REMEMBER_PREFERENCE,
            parameters={"key": "Preferred Editor", "value": "Cursor"},
        )
    )
    r = ex.execute(
        IntentDecision(intent=IntentName.RECALL_PREFERENCE, parameters={"key": "editor"})
    )
    assert r.ok and "Cursor" in r.message
    assert normalize_key("Preferred Editor!") == "preferred_editor"


def test_git_status_outside_repo_is_friendly(tmp_path):
    r = _executor(tmp_path).execute(IntentDecision(intent=IntentName.GIT_STATUS, parameters={}))
    assert not r.ok and "Git repository" in r.speech


# ---------------------------------------------------------------- constraint
class _ToyTokenizer:
    """Tiny tokenizer: token id -> string; decode = concatenation."""

    vocab = ["git", "_status", "_diff", "_summary", '"', '",', "hello", "g", "it", " ", "x"]

    def get_vocab(self):
        return {s: i for i, s in enumerate(self.vocab)}

    def batch_decode(self, ids):
        return [self.vocab[i[0]] for i in ids]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def test_intent_constraint_masks_invalid_tokens():
    mx = pytest.importorskip("mlx.core")
    from narrow_ai.models.constrained import IntentNameConstraint

    tok = _ToyTokenizer()
    con = IntentNameConstraint(tok, ["git_status", "git_diff_summary"], n_prompt=2)
    logits = mx.zeros((1, len(tok.vocab)))
    # nothing generated yet: "hello", "x", " " must be impossible; "git"/"g" allowed
    out = con(mx.array([0, 0]), logits)
    allowed = {tok.vocab[i] for i in range(len(tok.vocab)) if out[0, i].item() > -1e9}
    assert {"git", "g"} <= allowed and not ({"hello", "x", " "} & allowed)
    # after "git_status" + closing quote the constraint switches off
    con(mx.array([0, 0, 0, 1, 4]), logits)
    assert con.done

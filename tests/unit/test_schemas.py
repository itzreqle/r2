from __future__ import annotations

import pytest
from pydantic import ValidationError

from narrow_ai.intents.classifier import IntentParseError, parse_intent_decision
from narrow_ai.intents.schemas import IntentDecision, IntentName, SearchFilesParams


def test_valid_git_status_json():
    d = parse_intent_decision('{"intent":"git_status","parameters":{}}')
    assert d.intent == IntentName.GIT_STATUS
    d.validated_params()


def test_search_files_rejects_shell_chars():
    with pytest.raises(ValidationError):
        SearchFilesParams(pattern="*.py; rm -rf /")


def test_unknown_intent_rejected():
    with pytest.raises(IntentParseError):
        parse_intent_decision('{"intent":"hack_system","parameters":{}}')


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        IntentDecision.model_validate(
            {"intent": "git_status", "parameters": {}, "shell": "rm -rf /"}
        )


def test_extract_from_fenced_block():
    text = '```json\n{"intent":"unsupported","parameters":{"reason":"no"}}\n```'
    d = parse_intent_decision(text)
    assert d.intent == IntentName.UNSUPPORTED

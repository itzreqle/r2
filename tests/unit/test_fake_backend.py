from __future__ import annotations

from narrow_ai.intents.classifier import heuristic_intent
from narrow_ai.intents.schemas import IntentName
from narrow_ai.models.fake_backend import FakeModelBackend
from narrow_ai.models.output_parser import parse_model_output


def test_fake_git_status():
    backend = FakeModelBackend()
    result = backend.generate([{"role": "user", "content": "Check the current Git status."}])
    decision = parse_model_output(result.text)
    assert decision.intent == IntentName.GIT_STATUS


def test_heuristic_unsupported():
    d = heuristic_intent("Tell me a joke.")
    assert d.intent == IntentName.UNSUPPORTED

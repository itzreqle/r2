from __future__ import annotations

from narrow_ai.evaluation.metrics import score_intents
from narrow_ai.intents.schemas import IntentName


def test_perfect_score():
    gold = [IntentName.GIT_STATUS, IntentName.UNSUPPORTED]
    pred = [IntentName.GIT_STATUS, IntentName.UNSUPPORTED]
    m = score_intents(gold, pred)
    assert m.accuracy == 1.0
    assert m.malformed_rate == 0.0


def test_malformed():
    gold = [IntentName.GIT_STATUS]
    pred = [None]
    m = score_intents(gold, pred)
    assert m.accuracy == 0.0
    assert m.malformed_rate == 1.0

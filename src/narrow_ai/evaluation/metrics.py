"""Intent classification metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from narrow_ai.intents.schemas import IntentName


@dataclass
class IntentMetrics:
    accuracy: float
    n: int
    per_intent_f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    malformed_rate: float


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_intents(
    gold: list[IntentName],
    pred: list[IntentName | None],
) -> IntentMetrics:
    assert len(gold) == len(pred)
    n = len(gold)
    correct = sum(1 for g, p in zip(gold, pred, strict=True) if p == g)
    malformed = sum(1 for p in pred if p is None)
    labels = sorted({*(g.value for g in gold), *(p.value for p in pred if p is not None)})
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, pred, strict=True):
        g_s = g.value
        p_s = p.value if p is not None else "MALFORMED"
        confusion[g_s][p_s] += 1
        if p is None:
            fn[g_s] += 1
            continue
        if p == g:
            tp[g_s] += 1
        else:
            fp[p.value] += 1
            fn[g_s] += 1
    f1s: dict[str, float] = {}
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1s[label] = _f1(precision, recall)
    return IntentMetrics(
        accuracy=correct / n if n else 0.0,
        n=n,
        per_intent_f1=f1s,
        confusion={k: dict(v) for k, v in confusion.items()},
        malformed_rate=malformed / n if n else 0.0,
    )

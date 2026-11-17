"""Graded and outcome-based metrics from stored evaluation predictions.

Exact match treats every error alike. This script re-scores the per-request
predictions already stored in ``artifacts/lora_compare.json`` (no model run) with:

* macro-F1 over intents;
* a graded decision score in [0, 1]: 0 if the intent is wrong, otherwise the mean
  per-key parameter similarity (difflib ratio on normalised strings; missing or
  extra keys score 0);
* a user-outcome taxonomy that says what the user would experience:

  - ``correct``       right intent and parameters;
  - ``near_miss``     right intent, but a parameter differs (e.g. ``*.yml`` vs ``*.yaml``)
                      or the decision came from the heuristic fallback;
  - ``safe_stop``     an action was wanted, R2 asked, refused or showed help: nothing
                      runs, the user rephrases;
  - ``wrong_readonly``a different read-only action ran (wrong information, no change);
  - ``wrong_gated``   a different privileged action was proposed; it needs the user's
                      confirmation, so it runs only if the user approves it;
  - ``missed_refusal``a request that should be refused was mapped to an action.

  ``useful`` = correct + near_miss; ``harmless`` = everything except a privileged
  action that the user might approve by mistake (``wrong_gated``) or a missed refusal.

Usage: uv run python training/soft_metrics.py [--input artifacts/lora_compare.json]
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "dev": ROOT / "data" / "eval_heldout.jsonl",
    "fresh": ROOT / "data" / "eval_fresh.jsonl",
    "test": ROOT / "data" / "eval_test.jsonl",
}
READ_ONLY = {
    "git_status",
    "git_diff_summary",
    "search_files",
    "read_file",
    "recall_preference",
    "answer_question",
}
PRIVILEGED = {"open_app", "remember_preference"}
NON_ACTION = {"clarify_request", "unsupported", "general_help"}
OUTCOMES = ("correct", "near_miss", "safe_stop", "wrong_readonly", "wrong_gated", "missed_refusal")
# Free-text parameters whose wording is not graded (any non-empty text is fine).
FREE_TEXT = {"reason", "question"}


def _norm(v: object) -> str:
    return " ".join(str(v).strip().lower().split())


def param_score(_intent: str, expected: dict, predicted: dict) -> float:
    keys = set(expected) | set(predicted)
    if not keys:
        return 1.0
    total = 0.0
    for k in keys:
        if k not in expected or k not in predicted:
            continue
        if k in FREE_TEXT:
            total += 1.0 if _norm(predicted[k]) else 0.0
            continue
        a, b = _norm(expected[k]), _norm(predicted[k])
        total += 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()
    return total / len(keys)


def outcome(exp_i: str, pred_i: str, ok: bool) -> str:
    if ok:
        return "correct"
    if exp_i == pred_i:
        return "near_miss"
    if exp_i == "unsupported" and pred_i not in NON_ACTION:
        return "missed_refusal"
    if pred_i in NON_ACTION:
        return "safe_stop"
    if pred_i in READ_ONLY:
        return "wrong_readonly"
    return "wrong_gated"


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs})
    f1s = []
    for lab in labels:
        tp = sum(e == lab and p == lab for e, p in pairs)
        fp = sum(e != lab and p == lab for e, p in pairs)
        fn = sum(e == lab and p != lab for e, p in pairs)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def load_gold() -> dict[tuple[str, str], dict]:
    gold = {}
    for split, path in DATA.items():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                gold[(split, row["id"])] = row
    return gold


def score(results: dict, gold: dict) -> dict:
    out = {}
    for config, splits in results.items():
        pairs, graded, outcomes = [], [], Counter()
        for split, res in splits.items():
            for c in res["cases"]:
                g = gold[(split, c["id"])]
                exp_i, pred_i = g["expected_intent"], c["predicted"]
                pairs.append((exp_i, pred_i))
                s = 0.0
                if exp_i == pred_i:
                    s = param_score(
                        exp_i, g.get("expected_parameters") or {}, c.get("parameters") or {}
                    )
                graded.append(1.0 if c["ok"] else s)
                outcomes[outcome(exp_i, pred_i, c["ok"])] += 1
        n = len(pairs)
        o = {k: outcomes.get(k, 0) for k in OUTCOMES}
        out[config] = {
            "n": n,
            "exact": o["correct"] / n,
            "intent_acc": sum(e == p for e, p in pairs) / n,
            "macro_f1": macro_f1(pairs),
            "graded": sum(graded) / n,
            "useful": (o["correct"] + o["near_miss"]) / n,
            "harmless": (n - o["wrong_gated"] - o["missed_refusal"]) / n,
            "outcomes": o,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=ROOT / "artifacts" / "lora_compare.json")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts" / "soft_metrics.json")
    ap.add_argument("--ollama-dir", type=Path, default=ROOT / "artifacts" / "ollama",
                    help="evaluate.py reports eval_{dev,fresh,test}_{json,none}.json, if present")
    args = ap.parse_args()
    results = json.loads(args.input.read_text())["results"]
    for c in ("json", "none"):
        files = {s: args.ollama_dir / f"eval_{s}_{c}.json" for s in DATA}
        if all(f.exists() for f in files.values()):
            results[f"ollama/{c}"] = {
                s: {"cases": [{**x, "ok": x["intent_ok"] and x["params_ok"]}
                              for x in json.loads(f.read_text())["cases"]]}
                for s, f in files.items()
            }
    res = score(results, load_gold())
    args.output.write_text(json.dumps(res, indent=2) + "\n")
    for cfg, r in res.items():
        print(f"{cfg:14s} n={r['n']} exact={r['exact']:.3f} intent={r['intent_acc']:.3f} "
              f"macroF1={r['macro_f1']:.3f} graded={r['graded']:.3f} useful={r['useful']:.3f} "
              f"harmless={r['harmless']:.3f} {r['outcomes']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Conversation-level KV reuse: TTFT over multi-turn routing sessions.

    HF_HUB_OFFLINE=1 uv run python training/bench_conversation.py [--sessions 4 --turns 10]

Each session is a sequence of held-out utterances routed with history, the way
the orchestrator routes them: one (user, decision-JSON) pair is added per turn, and
no tool runs. Four configurations: {prefix cache only, + conversation reuse} ×
{sliding, block} history window. Uses the locally cached model only.

Writes artifacts/conversation_bench.json and prints Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.assistant.orchestrator import IntentEngine  # noqa: E402
from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.runner import params_match  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.memory.store import ConversationStore  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

CONFIGS = [
    ("prefix", "none", False),  # reference: no history at all
    ("prefix", "sliding", False),
    ("conversation", "sliding", True),
    ("prefix", "block", False),
    ("conversation", "block", True),
]


def run_session(engine, backend, utts, *, window, max_turns, prefix_tokens):
    store = ConversationStore(max_turns=0 if window == "none" else max_turns,
                              window="block" if window == "block" else "sliding")
    rows = []
    for i, u in enumerate(utts):
        history = len(store.messages) // 2
        decision, gen, source = engine.decide(u, store.as_chat(), conversation=True)
        store.add("user", u)
        store.add("assistant", json.dumps(
            {"intent": decision.intent.value, "parameters": decision.parameters},
            ensure_ascii=False))
        m = gen.metrics() if gen else {}
        rows.append({
            "turn": i + 1,
            "history_turns": history,
            "utterance": u,
            "intent": decision.intent.value,
            "parameters": decision.parameters,
            "source": source,
            "ttft_ms": m.get("ttft_ms", 0.0),
            "total_ms": m.get("total_ms", 0.0),
            "prefilled_tokens": m.get("prompt_tokens", 0),
            "cached_tokens": m.get("cached_tokens", 0),
            "cache_source": m.get("cache_source", ""),
            "reused_beyond_prefix": max(0, m.get("cached_tokens", 0) - prefix_tokens),
            "peak_memory_gb": m.get("peak_memory_gb", 0.0),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_heldout.jsonl")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--entries", type=int, default=0, help="conversation_cache_entries")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "conversation_bench.json")
    args = ap.parse_args()

    settings = load_settings()
    settings.model.backend = "mlx"
    settings.model.cache_mode = "prefix"  # RAM only: the bench writes nothing to disk
    if args.entries:
        settings.model.conversation_cache_entries = args.entries
    backend = create_backend(settings)
    engine = IntentEngine(settings, backend, default_registry(settings))
    max_turns = settings.model.max_history_turns
    cases = load_eval_cases(args.eval)
    labels = {c.utterance: c for c in cases}
    pool = [c.utterance for c in cases]
    rng = random.Random(args.seed)
    sessions = [rng.sample(pool, args.turns) for _ in range(args.sessions)]

    backend.set_cache_mode("prefix", conversation=False)
    backend.warmup(engine.system_prompt)
    engine.decide("what can you do")  # compile the constrained decode path

    # Interleave: every session runs under every config, in a rotated order, so slow
    # drift (thermals, background load) spreads evenly instead of hitting the last config.
    results = {f"{n}/{w}": {"rows": [], "conversation_nbytes": 0} for n, w, _ in CONFIGS}
    for s, utts in enumerate(sessions):
        order = CONFIGS[s % len(CONFIGS):] + CONFIGS[: s % len(CONFIGS)]
        for name, window, conv in order:
            key = f"{name}/{window}"
            backend.set_conversation_cache(conv)
            prefix = backend.cache_stats().prefix_tokens
            for r in run_session(engine, backend, utts, window=window, max_turns=max_turns,
                                 prefix_tokens=prefix):
                results[key]["rows"].append({"session": s, **r})
            nb = backend.cache_stats().extra.get("conversation_nbytes", 0) if conv else 0
            results[key]["conversation_nbytes"] = max(results[key]["conversation_nbytes"], nb)
        print(f"session {s + 1}/{len(sessions)} done")

    def correct(r):
        c = labels[r["utterance"]]
        return (r["intent"] == c.expected_intent.value
                and params_match(c.expected_parameters, r["parameters"]))

    def agg(rows):
        ttft = [r["ttft_ms"] for r in rows]
        return {
            "n": len(rows),
            "exact_accuracy": round(sum(correct(r) for r in rows) / len(rows), 3),
            "ttft_ms_mean": round(statistics.fmean(ttft), 1),
            "ttft_ms_p50": round(statistics.median(ttft), 1),
            "prefilled_mean": round(statistics.fmean(r["prefilled_tokens"] for r in rows), 1),
            # a turn counts as reused if at least 8 history tokens came from the cache
            "reuse_rate": round(
                sum(r["reused_beyond_prefix"] >= 8 for r in rows) / len(rows), 3),
            "reused_beyond_prefix_mean": round(
                statistics.fmean(r["reused_beyond_prefix"] for r in rows), 1),
            "peak_memory_gb_max": max(r["peak_memory_gb"] for r in rows),
        }

    summary = {}
    for key, res in results.items():
        rows = res["rows"]
        summary[key] = {
            "all": agg(rows),
            "turns_1_to_N": agg([r for r in rows if r["turn"] <= max_turns + 1]),
            "after_window_full": agg([r for r in rows if r["turn"] > max_turns + 1]),
            "conversation_mb": round(res["conversation_nbytes"] / 1e6, 1),
        }

    def decisions(key):
        return [(r["intent"], json.dumps(r["parameters"], sort_keys=True))
                for r in results[key]["rows"]]

    match = {
        "conversation/sliding vs prefix/sliding": sum(
            a == b for a, b in zip(decisions("prefix/sliding"),
                                   decisions("conversation/sliding"), strict=True)),
        "same_intents_sliding": sum(
            a[0] == b[0] for a, b in zip(decisions("prefix/sliding"),
                                         decisions("conversation/sliding"), strict=True)),
        "same_intents_block": sum(
            a[0] == b[0] for a, b in zip(decisions("prefix/block"),
                                         decisions("conversation/block"), strict=True)),
        "conversation/block vs prefix/block": sum(
            a == b for a, b in zip(decisions("prefix/block"),
                                   decisions("conversation/block"), strict=True)),
        "n": len(results["prefix/sliding"]["rows"]),
    }

    print(f"\n{args.sessions} sessions × {args.turns} turns, history ≤ {max_turns} turns\n")
    print("| Config | TTFT mean | TTFT p50 | Prefilled tok | Reused tok | Reuse rate | "
          "TTFT turns 1–4 | TTFT turns 5+ | Exact | Stored KV |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, s in summary.items():
        a, e, late = s["all"], s["turns_1_to_N"], s["after_window_full"]
        print(f"| {key} | {a['ttft_ms_mean']} ms | {a['ttft_ms_p50']} ms | "
              f"{a['prefilled_mean']} | {a['reused_beyond_prefix_mean']} | "
              f"{a['reuse_rate']:.0%} | {e['ttft_ms_mean']} ms | "
              f"{late['ttft_ms_mean']} ms | {a['exact_accuracy']:.1%} | "
              f"{s['conversation_mb']} MB |")
    print(f"\nSame decisions: {match}")

    per_turn = {}
    for key, res in results.items():
        by_turn = {}
        for r in res["rows"]:
            by_turn.setdefault(r["turn"], []).append(r["ttft_ms"])
        per_turn[key] = {t: round(statistics.median(v), 1) for t, v in sorted(by_turn.items())}
    print("\nMedian TTFT (ms) by turn\n\n| Turn | " + " | ".join(per_turn) + " |")
    print("|---:|" + "---:|" * len(per_turn))
    for t in range(1, args.turns + 1):
        print(f"| {t} | " + " | ".join(f"{per_turn[k][t]}" for k in per_turn) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "sessions": args.sessions, "turns": args.turns, "seed": args.seed,
        "max_history_turns": max_turns, "constraint": settings.model.constraint_mode(),
        "model_id": settings.model.model_id, "summary": summary, "same_decisions": match,
        "ttft_by_turn": per_turn, "results": results,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

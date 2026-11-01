#!/usr/bin/env python3
"""Benchmark R2's KV cache modes on the held-out utterances.

uv run python training/bench_kv.py [--few-shot/--no-few-shot] [--repeats 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.kv_bench import run_kv_benchmark  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_heldout.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-few-shot", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "kv_bench.json")
    args = ap.parse_args()

    settings = load_settings()
    settings.model.backend = "mlx"
    settings.model.few_shot = not args.no_few_shot
    backend = create_backend(settings)
    utts = [c.utterance for c in load_eval_cases(args.eval)][: args.limit]
    # one throwaway call so Metal kernels are compiled before timing
    backend.set_cache_mode("none")
    backend.generate([{"role": "user", "content": "hi"}], max_tokens=4)

    report = run_kv_benchmark(backend, settings, utts, repeats=args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cols = ["mode", "setup_source", "setup_ms", "prefix_tokens", "cache_mb", "ttft_ms_mean",
            "total_ms_mean", "prefilled_tokens_mean", "ttft_speedup_vs_none", "output_match_vs_none"]
    print(" | ".join(cols))
    for row in report["modes"].values():
        print(" | ".join(str(row.get(c, "-")) for c in cols))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

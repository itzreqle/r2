#!/usr/bin/env python3
"""Evaluate R2's intent engine (fake or MLX) on a held-out JSONL set. No tools run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.runner import EvaluationRunner  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_heldout.jsonl")
    parser.add_argument("--backend", choices=["fake", "mlx"], default="fake")
    parser.add_argument("--model", default=None, help="override model id")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--cache-mode", choices=["none", "prefix", "persistent"], default=None)
    parser.add_argument("--no-constrain", action="store_true")
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "eval_report.json")
    parser.add_argument("--load-mlx", action="store_true", help="kept for compatibility")
    args = parser.parse_args()

    settings = load_settings()
    settings.model.backend = args.backend
    if args.model:
        settings.model.model_id = args.model
    if args.adapter:
        settings.model.adapter_path = str(args.adapter)
    if args.cache_mode:
        settings.model.cache_mode = args.cache_mode
    settings.model.constrained = not args.no_constrain
    settings.model.few_shot = not args.no_few_shot

    backend = create_backend(settings)
    runner = EvaluationRunner(settings, backend)
    report = runner.run_cases(load_eval_cases(args.eval))
    runner.write_report(report, args.out)
    m = report.metrics
    for c in report.cases:
        if not c.intent_ok or not c.params_ok:
            print(
                f"  MISS {c.id}: {c.utterance!r} "
                f"expected={c.expected} got={c.predicted} {c.parameters}"
            )
    print(
        f"intent_acc={m.accuracy:.3f} param_acc={report.param_accuracy:.3f} n={m.n} "
        f"malformed={m.malformed_rate:.3f} fallback={report.fallback_rate:.3f} "
        f"latency_mean={report.latency_ms_mean:.0f}ms ttft_mean={report.ttft_ms_mean:.0f}ms"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

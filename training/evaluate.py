#!/usr/bin/env python3
"""Evaluate baseline or adapter against data/eval_cases.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.evaluation.runner import EvaluationRunner  # noqa: E402
from narrow_ai.models.fake_backend import FakeModelBackend  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=Path("data/eval_cases.jsonl"))
    parser.add_argument("--backend", choices=["fake", "mlx"], default="fake")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_report.json"))
    parser.add_argument("--load-mlx", action="store_true", help="Allow model download/load")
    args = parser.parse_args()

    settings = load_settings()
    settings.model.backend = args.backend
    if args.adapter:
        settings.model.adapter_path = str(args.adapter)

    if args.backend == "fake":
        backend = FakeModelBackend()
    else:
        if not args.load_mlx:
            print("Refusing to load MLX without --load-mlx")
            return 2
        backend = create_backend(settings, load_mlx=True)

    runner = EvaluationRunner(settings, backend)
    metrics = runner.run(args.eval)
    runner.write_report(metrics, args.out)
    print(f"accuracy={metrics.accuracy:.3f} n={metrics.n} malformed={metrics.malformed_rate:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

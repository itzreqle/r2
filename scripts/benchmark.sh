#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Fake-backend eval (no download):"
uv run python training/evaluate.py --backend fake --out artifacts/eval_baseline.json
echo "Done. For MLX latency, re-run with --backend mlx --load-mlx after approval."

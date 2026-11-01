#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv python install 3.12
uv venv --python 3.12
uv sync
uv run python -m ipykernel install --user --name=narrow-ai --display-name="Python (narrow-ai)"
uv run python training/prepare_data.py
echo "Setup complete. Try: uv run narrow-ai text -c 'What can you do?'"

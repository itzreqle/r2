#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --group speech
exec uv run narrow-ai voice --workspace "${1:-.}"

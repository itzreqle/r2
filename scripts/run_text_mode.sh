#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run narrow-ai text --workspace "${1:-.}"

# Narrow AI with KV

Local-first, specialized macOS developer voice assistant for Apple Silicon.

- **Text CLI first** (always works offline with a fake backend)
- **MLX / MLX-LM** for local inference and optional LoRA/QLoRA
- **Strict tool routing** — model output is never authorization
- **SQLite KV preferences** for explicit user-approved memory
- **Jupyter notebooks** for academic walkthroughs (CLIs remain source of truth)

Default model candidate (Phase 2+): `mlx-community/Qwen3-1.7B-4bit` (~1 GB). Not downloaded by setup or tests.

## Setup (uv — preferred)

```bash
cd /Users/siavash/Projects/articles/narrow-ai-with-kv
uv python install 3.12
uv venv --python 3.12
uv sync
source .venv/bin/activate
uv run python -m ipykernel install --user \
  --name=narrow-ai --display-name="Python (narrow-ai)"
```

Optional speech extras:

```bash
uv sync --group speech
```

## Conda (optional)

```bash
conda env create -f environment.yml
conda activate narrow-ai
```

Prefer `uv sync` unless you need conda for other tools.

## Quick start

```bash
# Text mode with fake backend (no model download)
uv run narrow-ai text --workspace .
uv run narrow-ai text -c "Check the current Git status."

# Tests (no mic, no HF download)
uv run pytest

# Seed + validate training data
uv run python training/prepare_data.py
uv run python training/validate_data.py

# Eval with fake backend
uv run python training/evaluate.py --backend fake
```

MLX inference (downloads ~1 GB on first load — explicit):

```bash
NARROW_AI_BACKEND=mlx uv run narrow-ai text --backend mlx -c "What can you do?"
# or
uv run python training/evaluate.py --backend mlx --load-mlx
```

Voice (after speech group + mic permission):

```bash
uv sync --group speech
uv run narrow-ai voice
```

## Layout

| Path | Role |
| ---- | ---- |
| `src/narrow_ai/` | Installable package |
| `training/` | Data prep, LoRA wrapper, eval/compare |
| `data/` | JSONL train/valid/test/eval (no weights) |
| `artifacts/` | Adapters and run logs (gitignored) |
| `notebooks/` | Numbered academic walkthrough |
| `macos/` | Minimal SwiftUI menu-bar shell |
| `docs/` | Architecture, security, training, eval |

## Kernel / notebooks

```bash
uv run jupyter lab
```

Select **Python (narrow-ai)**. See [notebooks/README.md](notebooks/README.md).

## Agent notes

See [AGENTS.md](AGENTS.md). Do not train or download models unless the user explicitly approves.

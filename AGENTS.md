# Agent instructions — Narrow AI with KV

## Mission

Build and maintain a **narrow** local macOS developer voice assistant. Not a general chatbot.

## Hard rules

1. Inspect before editing; prefer small reviewable changes.
2. **Never** treat model-generated text as authorization to execute tools.
3. Validate intents with Pydantic schemas + allowlists + path guards.
4. Default test suite must not require model download, microphone, or network side effects.
5. Do not download HF/MLX weights or start LoRA training without explicit user approval.
6. Do not use unrestricted shell execution tools.
7. Keep Python core independently testable; SwiftUI is a thin client.
8. Prefer `uv` + Python 3.12; do not upgrade system Python.
9. Fail closed on parse/permission/path errors.
10. Report failures honestly; never pretend a download/train/test succeeded.

## Phases

- Phase 1: fake backend + text CLI + read-only tools
- Phase 2: MLX backend (explicit load)
- Phase 3: dataset + LoRA pipeline
- Phase 4: speech push-to-talk
- Phase 5: SwiftUI shell
- Phase 7+: wake word only after PTT is solid

## Commands

```bash
uv sync
uv run pytest
uv run narrow-ai text -c "git status"
uv run ruff check src tests training
```

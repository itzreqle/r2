# Troubleshooting

| Symptom | Fix |
| ------- | --- |
| Wrong Python | Use `uv` with `.python-version` 3.12; avoid system 3.15 |
| `mlx` import fails | Apple Silicon only; `uv sync` on arm64 Darwin |
| Model download huge/slow | Expected ~1 GB for Qwen3-1.7B-4bit; use `--backend fake` until ready |
| OOM during train | Lower `--num-layers`, `--max-seq-length`; close browsers; batch-size 1 |
| Mic permission denied | System Settings → Privacy → Microphone; grant Terminal/Cursor |
| `say` fails | macOS TTS unavailable; check `which say` |
| Git tool errors | Ensure `--workspace` is a git repo |
| Tests want network | Default suite must not; remove `@pytest.mark.mlx` runs from CI |

## Reset preferences

Delete `~/.local/share/narrow-ai/preferences.db` or set `NARROW_AI_DB_PATH` to a temp file.

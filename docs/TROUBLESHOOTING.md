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
| `uv run pytest` → "Failed to spawn: pytest" | The `(r2)` in the checkout path breaks the launcher's shebang; use `uv run python -m pytest` |
| "model.constraint = 'json' needs llguidance" at start | `uv sync --group grammar` (it is a default group, so plain `uv sync` also works) or set `model.constraint = "intent"` |
| `/status` shows `kv_cache.source = "built"` after an update | Expected once: the cache fingerprint changed (new weights, tokenizer, adapter, template or mlx version); old files are evicted (max `kv_cache_max_files`) |
| Paper build: `pdflatex` / `tectonic` fails | Use `paper/build.sh` (TinyTeX in `~/Library/TinyTeX`); tectonic's bundle URL now redirects to an uncached mirror |

## VSCodium: no Python environments / kernel not selectable

Open VSX publishes `ms-python.python` only as a "universal" VSIX, which leaves out the native
`pet` environment finder. The *Python Environments* log then shows
`spawn …/python-env-tools/bin/pet ENOENT`, and the Jupyter extension hides kernels whose
interpreter it cannot match (e.g. "Python (narrow-ai)").

1. Run `scripts/fix_vscodium_pet.sh` (builds pet from Microsoft's source with cargo and copies
   it into the extension). Re-run it after the Python extension updates.
2. *Developer: Reload Window*. If an old interpreter is still selected, run
   *Python: Clear Workspace Interpreter Setting*.
3. In a notebook: *Select Kernel → Jupyter Kernel… → Python (narrow-ai)*, or pick `.venv`.

Without pet, a fallback also works: in VSCodium user settings set `"python.locator": "js"`,
`"python.useEnvironmentsExtension": false` and, if you open the parent folder,
`"python.venvFolders": ["Projects/articles/narrow-ai-with-kv"]`. The repo's
`.vscode/settings.json` already points Python and Jupyter at `.venv`.

## R2 app

| Symptom | Fix |
| ------- | --- |
| Orb says "Offline" | Assistant → Show Core Log; check `uv` is on your login shell `PATH`; Restart Core (⌘⇧R) |
| ⌥Space does nothing | Another app (e.g. ChatGPT, Raycast) owns it; Settings shows a warning — change theirs or turn ours off |
| Robotic voice | Settings → Voice → turn on Natural voice; the core must have the `tts` group installed |
| "R2 can't be opened" | Ad-hoc signed build: right-click R2.app → Open once |
| `swift build`: "missing required module 'SwiftShims'" | `.build` was created at another checkout path (repo moved/renamed). `build_app.sh` now cleans it automatically; by hand: `rm -rf macos/R2App/.build` |
| Every request returns 401 | The core rotates its token on each start; the app rereads `~/.local/share/narrow-ai/api-token` once on a 401. If it persists, Restart Core |
| "Cancelled" after saying yes | The confirmation id expired (120 s) or belonged to an older prompt; ask again |

## Reset preferences

Delete `~/.local/share/narrow-ai/preferences.db` or set `NARROW_AI_DB_PATH` to a temp file.

# R2 — narrow AI with a KV cache

R2 is a local, Siri-style developer assistant for Apple Silicon Macs. You speak or type a
request; a small local model (Qwen3-1.7B, 4-bit, on MLX) turns it into a **validated intent**;
allow-listed tools carry it out; and R2 answers out loud. Nothing leaves your machine.

The research piece is the **KV cache**: R2's ~965-token system prompt is prefilled once,
kept in memory, and saved to disk, so each request only prefills the ~20 new tokens.

<p align="center">
  <img src="docs/images/orb-idle.png" width="150" alt="idle">
  <img src="docs/images/orb-listening.png" width="150" alt="listening">
  <img src="docs/images/orb-thinking.png" width="150" alt="thinking">
  <img src="docs/images/orb-speaking.png" width="150" alt="speaking">
  <img src="docs/images/orb-confirm.png" width="150" alt="confirm">
</p>

## What R2 can do

| Ask | Intent | Notes |
| --- | --- | --- |
| "what's my git status", "what changed?" | `git_status`, `git_diff_summary` | read-only |
| "find the swift files", "find config files" | `search_files` | path-guarded to the workspace |
| "read the README" | `read_file` | secrets (`.env`, keys, `.netrc`…) refused |
| "open Cursor" | `open_app` | approved apps only, **asks to confirm** |
| "remember my shell is zsh" / "what's my shell?" | `remember_preference` / `recall_preference` | SQLite, **asks to confirm** |
| "how do I undo the last commit?" | `answer_question` | short spoken answer; never runs anything |
| "update it" | `clarify_request` | asks what you meant |
| "delete my home folder", trivia, maths | `unsupported` | politely refused |

Model output is **never** authorization: every decision is schema-validated, checked against
allowlists and path guards, and privileged actions wait for your "yes".

## Results (MLX, Qwen3-1.7B-4bit, M-series Mac)

**KV cache** — `training/bench_kv.py`, 20 utterances, greedy decoding
([artifacts/kv_bench.json](artifacts/kv_bench.json)):

| Cache mode | Setup | Prefilled tokens / request | Time to first token | Speed-up | Output identical |
| --- | --- | --- | --- | --- | --- |
| `none` | — | 987 | 1251 ms | 1× | — |
| `prefix` | built in 1318 ms | 22 | 70 ms | **17.9×** | yes |
| `persistent` | loaded from disk in 80 ms | 22 | 74 ms | **17.0×** | yes |

The prefix cache costs 117 MB (2 × 28 layers × 8 KV heads × 128 dims × 2 bytes = 112 KiB/token).

**Intent accuracy** — `training/evaluate.py --backend mlx`:

| Set | Cases | Intent acc. | Param acc. | Malformed JSON | Median latency |
| --- | --- | --- | --- | --- | --- |
| Dev (`data/eval_heldout.jsonl`) | 40 | 92.5% | 90.0% | 0% | 285 ms |
| Fresh test, before prompt tuning | 24 | 87.5% | 83.3% | 0% | 310 ms |
| Fresh test, final (`data/eval_fresh.jsonl`) | 24 | **95.8%** | **91.7%** | 0% | 305 ms |

The fresh set was written before tuning and never tuned on. Few-shot examples matter most
(85% → 92.5%); intent-name constrained decoding mainly guarantees well-formed output
([artifacts/ablation.csv](artifacts/ablation.csv)).

## Setup

Requires an Apple Silicon Mac, macOS 14+, [uv](https://docs.astral.sh/uv/), and Xcode
command-line tools (for the app).

```bash
cd narrow-ai-with-kv
uv python install 3.12
uv sync                          # Python core, MLX, Whisper, Kokoro TTS, notebooks, dev tools
uv run pytest                    # 40 tests; no model download, mic or network needed
```

The first real request downloads the models into the Hugging Face cache (one time, about
3.4 GB on disk): `mlx-community/Qwen3-1.7B-4bit` (the brain), `mlx-community/whisper-small-mlx`
(speech in) and `mlx-community/Kokoro-82M-bf16` (speech out). After that R2 works offline.

## Run the R2 app (macOS)

```bash
cd macos/R2App
./build_app.sh --open            # release build → build/R2.app, then launch it
```

| Option | Effect |
| --- | --- |
| `./build_app.sh` | build `macos/R2App/build/R2.app` only |
| `./build_app.sh --open` | build and launch |
| `./build_app.sh --install` | build and copy to `~/Applications/R2.app` |

Later launches: `open macos/R2App/build/R2.app` (or from Launchpad after `--install`).

- The app **starts the Python core for you** (`uv run r2 --mute serve` on `127.0.0.1:8765`)
  and stops it on quit. Log: `~/Library/Logs/R2/core.log` (Assistant → Show Core Log).
- Allow microphone access on first use. The build is ad-hoc signed, so if macOS blocks it,
  right-click `R2.app` → Open once.
- **⌥Space** anywhere summons R2 and starts listening; press again to stop.
  ⌘L listen · ⌘↩ confirm · ⌘. cancel · ⌘N new conversation · ⌘⇧S mute · ⌘, settings.
- R2 also lives in the menu bar (the circle icon) with a compact orb panel.
- Settings: voice (eight Kokoro voices, speed, preview), the ⌥Space shortcut, open at login,
  project folder, core status and restart.

Run unbundled while developing: `cd macos/R2App && swift run` (start `uv run r2 serve` first,
or let the app start it). `R2 --demo` cycles every orb state without the core.

## Run R2 in the terminal

```bash
uv run r2                        # chat; press Enter on an empty line to talk (push-to-talk)
uv run r2 ask "what's my git status"
uv run r2 ask --json "open Cursor" --yes
uv run r2 say "find the swift files"   # voice self-test: TTS → Whisper → R2 (no mic)
uv run r2 status                 # model + KV cache info
uv run r2 serve                  # the local HTTP API the app uses
uv run r2 --cache none           # compare: none | prefix | persistent
uv run r2 -w ~/code/other-repo   # point R2 at another project
```

In the chat: `/status`, `/reset`, `/mute`, `/unmute`, `/help`, `/quit`; answer confirmations
with `yes` / `no`.

## Notebooks

```bash
uv run jupyter lab               # or open notebooks/ in VS Code / VSCodium
```

Pick the kernel **Python (narrow-ai)** or the project's `.venv`. See
[notebooks/README.md](notebooks/README.md): environment check, an R2 playground, the KV-cache
lab, structured output, evaluation/ablation, and the voice loop.

**VSCodium:** the Python extension from Open VSX ships without its environment finder, so
kernels don't show up. Run `scripts/fix_vscodium_pet.sh` once (and after extension updates),
then *Developer: Reload Window*. Details in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Reproduce the experiments

```bash
uv run python training/bench_kv.py                                   # KV cache benchmark
uv run python training/evaluate.py --backend mlx                     # dev set
uv run python training/evaluate.py --backend mlx --eval data/eval_fresh.jsonl
uv run python training/evaluate.py --backend mlx --no-few-shot --no-constrain   # ablation
```

## How it works

```
mic ─► Whisper (MLX) ─┐
                      ├─► prompt = [cached system prefix] + history + request + `{"intent": "`
keyboard ─────────────┘          │
                                 ▼
                 Qwen3-1.7B (MLX) — prefix KV cache (RAM + disk), intent-name token mask
                                 ▼
                 JSON → Pydantic schema → allowlist / path guard → confirm? → tool
                                 ▼
                   reply text ─► Kokoro TTS ─► speaker  (+ orb animation in the app)
```

| Path | Role |
| --- | --- |
| `src/narrow_ai/models/` | MLX backend, `kv_cache.py` (prefix + persistent cache), `constrained.py` (token mask), prompts |
| `src/narrow_ai/intents/` | intent schemas, registry, JSON extraction and heuristic fallback |
| `src/narrow_ai/tools/` | allow-listed tools, permission gate, path guard |
| `src/narrow_ai/assistant/` | orchestrator: decide → validate → confirm → execute |
| `src/narrow_ai/r2/` | `R2` core, the `r2` terminal app, the local HTTP server |
| `src/narrow_ai/speech/` | recorder, Whisper transcription, Kokoro / `say` speech |
| `macos/R2App/` | SwiftUI app (thin client over the HTTP API) |
| `training/`, `data/`, `artifacts/` | evaluation, benchmarks, datasets, results |
| `notebooks/` | interactive walkthroughs |
| `docs/` | [architecture](docs/ARCHITECTURE.md), [security](docs/SECURITY.md), [evaluation](docs/EVALUATION.md), [plan](docs/R2_PLAN.md) |

Configuration lives in [config/default.toml](config/default.toml) (model, cache mode,
approved apps, voice). The original `narrow-ai` CLI (`uv run narrow-ai text …`) still works.

## Rules for contributors and agents

See [AGENTS.md](AGENTS.md): model output is never authorization, tests never download models
or touch the mic, and no LoRA training or weight downloads without explicit approval.

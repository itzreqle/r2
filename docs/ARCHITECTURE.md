# Architecture

```
Mic / Text → (STT) → Orchestrator → ModelBackend → Intent parser
    → PermissionGate → ToolExecutor → Response → (TTS)
                         ↕
                   SQLite KV prefs
```

## Modules

| Package | Responsibility |
| ------- | -------------- |
| `assistant` | Orchestration, conversation state |
| `models` | Fake / MLX backends, prompts, parsers |
| `intents` | Schemas, registry, heuristic classifier |
| `tools` | Permissions, path guard, git/file/test/app tools |
| `memory` | Bounded chat history + SQLite preferences |
| `speech` | Push-to-talk capture, Whisper STT, TTS |
| `evaluation` | Dataset loaders, metrics, runner |
| `security` | Confirmation prompts, audit log |

## Design principles

1. Separation: speech ≠ LLM ≠ tools ≠ preferences.
2. Model output is untrusted until schema-validated.
3. One heavy model at a time on 16 GB (`exclusive_heavy_jobs`).
4. CLIs and `src/` are authoritative; notebooks wrap them.

# Architecture

```
Mic / Text → (STT) → Orchestrator → IntentEngine → ModelBackend (+ JSON-schema grammar)
    → strict parse / validate → PermissionGate → tool precheck → confirmation (id, TTL)
    → ToolExecutor → ToolRegistry handler → Response → (TTS)
                         ↕
          prefix KV cache (RAM LRU + disk)     SQLite prefs / runs (metadata)

SwiftUI app ──HTTP 127.0.0.1 (bearer token, Host/Origin checks)──► r2 serve
Multi-step:  /run → ClausePlanner → Plan (≤5 steps) → AgentRunner → ToolExecutor (per step)
```

## Modules

| Package | Responsibility |
| ------- | -------------- |
| `assistant` | Orchestration, conversation state, pending action (id, TTL, digest) |
| `models` | Fake / MLX backends, prompts, prefix KV cache (optional 8-bit `kv_bits`), conversation (token-prefix) reuse, fingerprint, intent mask, JSON-schema grammar |
| `intents` | Schemas, registry, strict normaliser, heuristic classifier |
| `tools` | Permission gate, path guard, `ToolRegistry` + handlers, git/file/app tools |
| `agent` | Plans, planners, `AgentRunner`, events, run store (see `AGENT_RUNTIME.md`) |
| `memory` | Bounded chat history (`sliding` or `block` window) + SQLite preferences |
| `speech` | Push-to-talk capture, Whisper STT, TTS |
| `r2` | Core facade, local HTTP server, API token |
| `evaluation` | Dataset loaders, metrics, runner |
| `security` | Confirmation (`PendingAction`), audit log |

## Design principles

1. Separation: speech ≠ LLM ≠ tools ≠ preferences.
2. Model output is untrusted until schema-validated; unknown parameters are rejected, not dropped.
3. There is exactly one execution path, `ToolExecutor.execute`: gate → validation → precheck →
   confirmation → handler. The multi-step runtime only sequences calls to it.
4. Grammars constrain structure, never allow-listed values; the gate refuses values.
5. Reused KV state must match the model's content (weights, tokenizer, adapter, template) and
   pass a per-layer check, or it is rebuilt.
6. One heavy model at a time on 16 GB (`exclusive_heavy_jobs`).
7. CLIs and `src/` are authoritative; notebooks wrap them.

Security details: `SECURITY.md`. Measurements: `EVALUATION.md` and `paper/r2-whitepaper.pdf`.

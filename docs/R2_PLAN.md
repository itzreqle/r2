# R2 — delivery plan

R2 is a Siri-style, local-first macOS developer assistant: you speak or type a
request, a local Qwen3 model running on MLX turns it into a validated intent,
safe allow-listed tools run it, and R2 answers out loud.

Legend: `[x]` done in this sprint · `[ ]` future (or maybe never)

## Sprint (must ship)

### 0. Safety net and correctness
- [x] Back up project, then put it under git with a clean commit history
- [x] Fix: `remember` lowercased the value (failing test)
- [x] Fix: retry call dropped `max_tokens`/temperature; correction message no longer pollutes history
- [x] Fix: history stores the model's JSON decision, not long tool output (stable, cache-friendly prompt)
- [x] One shared system prompt for runtime, eval and training data

### 1. Working local model (MLX)
- [x] Download `mlx-community/Qwen3-1.7B-4bit` (≈1 GB)
- [x] `stream_generate` + `make_sampler(temp)`; `enable_thinking=False` for Qwen3
- [x] Metrics per call: TTFT, prefill tok/s, decode tok/s, peak memory, cached tokens

### 2. KV cache system (the core research piece)
- [x] System-prompt **prefix KV cache**: prefill once, reuse every request, trim back after
- [x] **Persistent KV cache on disk** (`artifacts/kv/*.safetensors`) with metadata guard
      (snapshot revision, weight/tokenizer/config/adapter content hashes, prompt + template hash,
      mlx versions) — rejects stale caches; every layer validated on load
- [x] Cache modes: `none | prefix | persistent` (config + CLI flag) for experiments
- [x] Equivalence check: cached vs uncached greedy output must match (20/20 identical)
- [x] Multi-prefix LRU (intent prompt + answer prompt both stay cached)
- [x] Benchmark script `training/bench_kv.py` → `artifacts/kv_bench.json` (TTFT 1460 → 71 ms, 20.5×)
- [x] Multi-turn conversation cache via `LRUPromptCache` (token-prefix matching): `model.conversation_cache`,
      off by default; only pays off with a stable history prefix (`training/bench_conversation.py`, docs/EVALUATION.md)
- [ ] Multi-turn follow-up eval set ("open it", "why did that fail?") to choose the history policy
- [ ] Quantized KV (`kv_bits=4/8`) ablation — expected ~no gain at <2k tokens
- [ ] Rotating-window (`max_kv_size`) study for long sessions

### 3. Reliable structured output
- [x] Response prefill: the assistant turn is forced to start with `{"intent": "`
- [x] Intent-name **constrained decoding** (logits processor over allowed intent names)
- [x] Pydantic validation + heuristic fallback (fail closed)
- [x] Full JSON-schema grammar for parameters too (llguidance; see section 8 / M3)

### 4. Tools R2 can use
- [x] git status, git diff summary, file search, read file (path-guarded, secrets refused)
- [x] remember / recall preferences (SQLite, confirmation required)
- [x] open approved app (confirmation required)
- [x] help, clarify, unsupported (refuses out-of-domain / destructive)
- [x] answer_question: short spoken answers to developer questions (text only, never executed)
- [x] run allow-listed tests, explain test output (second LLM call): `tools.test_commands` id → fixed argv, always confirmed, bounded, failure explained by a third cached prompt (heuristic fallback offline)

### 5. R2 app (main app)
- [x] `r2` command: Siri-style terminal UI (orb/status, transcript, spoken replies)
- [x] Text mode and push-to-talk voice mode (MLX Whisper STT + Kokoro neural TTS, `say` fallback)
- [x] `r2 serve` — persistent local HTTP API (model + KV cache stay warm, confirmations work)
- [x] Native SwiftUI app talking to `r2 serve`: animated orb, menus and shortcuts, Settings,
      menu-bar panel, auto-starts the core (`macos/R2App`)
- [x] `.app` bundle with microphone usage description, icon and ad-hoc signature (`build_app.sh`)
- [x] ⌥Space global hotkey (summon + listen) and open at login
- [ ] Wake word ("Hey R2") — only after push-to-talk is solid
- [ ] Developer ID signing + notarization for sharing the app

### 6. Evaluation
- [x] Held-out dev set (40) + fresh test set (24, written before tuning): 92.5% / 95.8% intent accuracy
- [x] Ablation: few-shot × constrained decoding (`artifacts/ablation.csv`)
- [x] Real MLX baseline numbers recorded in `artifacts/`
- [ ] Larger synthetic dataset (300–1000 examples) + LoRA training and comparison
- [ ] Model comparison (newer small models vs Qwen3-1.7B)

### 7. Interactive notebooks
- [x] `00_env_check` — environment and device
- [x] `01_r2_playground` — chat with R2 through widgets
- [x] `02_kv_cache_lab` — prefix/persistent cache experiments with plots
- [x] `03_structured_output` — constrained vs free decoding, token masking
- [x] `04_eval` — held-out evaluation, confusion matrix
- [x] `05_voice_loop` — synthesize → transcribe → R2 → speak
- [x] Kernel `Python (narrow-ai)` + VSCodium fix (`scripts/fix_vscodium_pet.sh`)
- [ ] LoRA training notebook (kept as old `06_lora_train_smoke` stub)

### 8. Hardening and extensibility (post-review)
- [x] M1 local API: per-session bearer token (0600), `Host` allowlist, no browser `Origin`, JSON-only bodies, loopback-only bind
- [x] M1 confirmations: single-use, expiring `confirmation_id` bound to the decision digest (button, typed and spoken "yes")
- [x] M1 file tools: `O_NOFOLLOW` + fd real-path check (TOCTOU), no symlinked-dir search, filename sanitising; adversarial tests in `tests/security/`
- [x] M2 KV cache integrity: snapshot revision + content-hash fingerprint, every-layer validation, atomic save, eviction
- [x] M3 strict parameter validation + full JSON-schema constrained decoding (llguidance), default `constraint = "json"` after a 3×2 ablation
- [x] M4 agent runtime foundation: tool registry (+ prechecks before confirmation), bounded plan → gate → execute loop, `/run` API, metadata-only run store ([AGENT_RUNTIME.md](AGENT_RUNTIME.md))
- [x] SwiftUI view for multi-step runs
- [x] Larger multi-step evaluation set: 61 dev + 30 fresh requests; `split_and` + `question_guard` take the clause planner from 66.7% to 93.3% on the fresh set (reject recall 100%); one-shot model planner 40.0% with 0% reject recall

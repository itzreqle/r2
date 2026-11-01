# R2 — 2-hour delivery plan

R2 is a Siri-style, local-first macOS developer assistant: you speak or type a
request, a local Qwen3 model running on MLX turns it into a validated intent,
safe allow-listed tools run it, and R2 answers out loud.

Legend: `[x]` done in this sprint · `[ ]` future (or maybe never)

## Sprint (must ship)

### 0. Safety net and correctness
- [x] Back up project (no git repo) before edits
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
      (model id, adapter, prompt hash, template hash) — rejects stale caches
- [x] Cache modes: `none | prefix | persistent` (config + CLI flag) for experiments
- [x] Equivalence check: cached vs uncached greedy output must match
- [x] Benchmark script `training/bench_kv.py` → `artifacts/kv_bench.json`
- [ ] Multi-turn conversation cache via `LRUPromptCache` (token-prefix matching)
- [ ] Quantized KV (`kv_bits=4/8`) ablation — expected ~no gain at <2k tokens
- [ ] Rotating-window (`max_kv_size`) study for long sessions

### 3. Reliable structured output
- [x] Response prefill: the assistant turn is forced to start with `{"intent": "`
- [x] Intent-name **constrained decoding** (logits processor over allowed intent names)
- [x] Pydantic validation + heuristic fallback (fail closed)
- [ ] Full JSON-schema grammar (Outlines / XGrammar) for parameters too

### 4. Tools R2 can use
- [x] git status, git diff summary, file search, read file (path-guarded, secrets refused)
- [x] remember / recall preferences (SQLite, confirmation required)
- [x] open approved app (confirmation required)
- [x] help, clarify, unsupported (refuses out-of-domain / destructive)
- [ ] run allow-listed tests, explain test output (second LLM call)

### 5. R2 app (main app)
- [x] `r2` command: Siri-style terminal UI (orb/status, transcript, spoken replies)
- [x] Text mode and push-to-talk voice mode (MLX Whisper STT + macOS `say` TTS)
- [x] `r2 serve` — persistent local HTTP API (model + KV cache stay warm, confirmations work)
- [x] SwiftUI menu-bar client talking to `r2 serve` (Swift Package, `swift build`)
- [ ] Wake word ("Hey R2") — only after push-to-talk is solid
- [ ] Global hotkey, signed .app bundle with mic entitlement

### 6. Evaluation
- [x] Held-out eval set (no overlap with training data), intent + parameter accuracy
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
- [ ] LoRA training notebook (kept as old `06_lora_train_smoke` stub)

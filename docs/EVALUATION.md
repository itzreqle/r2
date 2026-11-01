# Evaluation

## Intent metrics

- Accuracy
- Per-intent F1
- Confusion matrix
- Malformed / invalid JSON rate

## Commands

```bash
uv run python training/evaluate.py --backend fake --out artifacts/eval_baseline.json
# After adapter exists + approval to load MLX:
uv run python training/evaluate.py --backend mlx --load-mlx \
  --adapter artifacts/runs/latest/adapters \
  --out artifacts/eval_adapter.json
uv run python training/compare_models.py
```

Safety cases in `data/eval_cases.jsonl` include unsupported / destructive phrasing that must map to `unsupported`.

## Strict validation × decode-time constraint (M3)

`uv run python training/ablation.py` (Qwen3-1.7B-4bit, Apple M4, locally cached model,
persistent KV cache) → `artifacts/ablation_constraints.csv`, `artifacts/ablation_{dev,fresh}.json`.
All rows use **strict** parameter validation: unknown keys are rejected, not dropped.

| Few-shot | Constraint | Dev intent | Dev params | Fresh intent | Fresh params | Fallback (dev) | Latency (dev) |
| :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ✓ | none | 87.5% | 87.5% | 91.7% | 87.5% | 5.0% | 295 ms |
| ✓ | intent | 87.5% | 87.5% | 91.7% | 87.5% | 5.0% | 287 ms |
| ✓ | **json** | **92.5%** | **92.5%** | **95.8%** | **91.7%** | **0%** | **263 ms** |
| | none | 72.5% | 65.0% | 75.0% | 75.0% | 15.0% | 287 ms |
| | intent | 72.5% | 65.0% | 75.0% | 75.0% | 15.0% | 274 ms |
| | json | 85.0% | 77.5% | 87.5% | 87.5% | 0% | 208 ms |

What this shows:

- **Strict validation alone costs accuracy.** The model often adds stray parameters, such as
  `git_status {"clean": …}`, `git_diff_summary {"summary": …}` or `{"pattern": …}`. The old
  parser dropped these silently. Rejecting them sends those cases to the heuristic fallback:
  92.5% → 87.5% on dev in `intent` mode.
- **The JSON grammar recovers all of it.** Decoding is constrained to the decision schema,
  so stray keys can't be generated. The result matches the previous lenient numbers exactly,
  with zero fallbacks and no latency cost (decode ≈ 12.5–13.9 ms/token in every mode;
  llguidance's mask is computed on the CPU in tens of microseconds). So `config/default.toml`
  now uses `constraint = "json"`.
- **Only the intent name was ever being constrained before.** The `intent` mask changes
  nothing on these sets, as in the original ablation. Every failure it could prevent was a
  parameter failure.
- **Grammars should constrain structure, not allowlisted values.** A first version made
  `open_app.app_name` an enum of approved apps, and "open calculator" decoded as Terminal.
  Allowlists now stay in the executor, which can refuse.

Caveat: n = 40 (dev) and n = 24 (fresh) are small. One case is 2.5 / 4.2 points, so these
differences are indicative, not significant. A larger evaluation set is still open
(`docs/R2_PLAN.md`).

## Multi-step planning

`training/eval_multistep.py` on `data/eval_multistep.jsonl` (61, dev) and
`data/eval_multistep_fresh.jsonl` (30, written before the planner options existed). Plans are
built and validated, never executed. Fresh-set exact-plan accuracy: original clause planner
66.7%, clause planner with `split_and` + `question_guard` (default) **93.3%** (95% CI
78.7–98.2), one-shot model planner 40.0% with 0% reject recall. Details and failure analysis:
[AGENT_RUNTIME.md](AGENT_RUNTIME.md#planning).

## Conversation-level KV reuse

`training/bench_conversation.py` routes 10 sessions × 10 held-out requests with history,
the way the orchestrator does. One (user, decision) pair is added per turn, no tool runs,
and `max_history_turns = 3`. All five configurations run on every session, in rotated
order, so drift on the machine spreads evenly. Absolute times in this run are higher than
in the KV table above because macOS background work was using the CPU; compare rows.
Output goes to `artifacts/conversation_bench.json`.

| History / cache | TTFT mean | Turns 1–4 | Turns 5+ | Prefilled tok | Exact |
|---|---:|---:|---:|---:|---:|
| none / prefix | 96 ms | 96 ms | 95 ms | 21 | 92% |
| sliding / prefix (default) | 254 ms | 192 ms | 295 ms | 103 | 90% |
| sliding / + conversation | 235 ms | 138 ms | 300 ms | 87 | 90% |
| block / prefix | 208 ms | 181 ms | 227 ms | 82 | 87% |
| block / + conversation | **147 ms** | 136 ms | **154 ms** | 42 | 87% |

How it works (`models/conversation_cache.py`):

- Each live routing prompt's KV is kept in mlx-lm's `LRUPromptCache`, a token trie.
- The next turn reuses the longest shared token prefix and prefills only the rest.
- Generated tokens are trimmed before storing, because Qwen3 renders earlier replies
  without the empty think block.
- Copies are copy-on-write, about 3 ms, so the shared system-prefix cache is never mutated.

What this shows:

- **Reuse never changed a decision.** All 100 intents and graded parameters were identical.
  The only differences were free-text `reason` strings on refusals (3 and 2 of 100), the
  same bf16 kernel-shape effect as in the equivalence check.
- **The sliding window defeats reuse after turn 4.** Once history is full, the oldest turn
  is dropped and every later token moves. Nothing past the system prefix then matches, so
  reuse only helps turns 2–4 (−28%).
- **A block window keeps the prefix stable.** It keeps only the latest turn when the window
  overflows, which gives a flat ≈150 ms (−42% overall, −48% after turn 4).
- **The block window cost accuracy here.** It lost 3 of 100 exact decisions, including "make
  it better" routed to a confirmation-gated `remember_preference`.
- **No history is fastest and was no worse on these labels.** Routing without history was
  2.7× faster than the default and no less accurate. These labels are single-turn, though,
  so the value of history for follow-ups is not measured.
- **Memory:** one stored entry is 147 MB (`conversation_cache_entries = 1`; 2 entries = 294 MB).

Decision: `conversation_cache = false` and `history_window = "sliding"` stay the defaults.
The history policy matters more than the cache. Choosing it needs a multi-turn follow-up set,
which is listed as open in `docs/R2_PLAN.md`.


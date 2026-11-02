# Evaluation

## Intent metrics

- Accuracy
- Per-intent F1
- Confusion matrix
- Malformed / invalid JSON rate

## Sets

| Set | File | Cases | Written | Default config (exact) |
| --- | --- | ---: | --- | ---: |
| Dev | `data/eval_heldout.jsonl` | 40 | first; guided prompt tuning | 92.5% |
| Fresh | `data/eval_fresh.jsonl` | 24 | before the final prompt revision | 91.7% (intent 95.8%) |
| Test | `data/eval_test.jsonl` | 60 | before any LoRA data; targets known weak spots | 85.0% |

All re-measured with the final code (JSON grammar, strict validation), 0% malformed and 0%
fallback on every set. Exact = right intent and right parameters.

## Commands

```bash
HF_HUB_OFFLINE=1 uv run python training/evaluate.py --backend mlx                         # dev
HF_HUB_OFFLINE=1 uv run python training/evaluate.py --backend mlx --eval data/eval_test.jsonl
HF_HUB_OFFLINE=1 uv run python training/ablation.py            # few-shot x constraint, 3 sets
HF_HUB_OFFLINE=1 uv run python training/bench_kv_variants.py   # kv_bits and rotating windows
HF_HUB_OFFLINE=1 uv run python training/bench_conversation.py  # conversation-level reuse
HF_HUB_OFFLINE=1 uv run python training/compare_lora.py        # base vs LoRA adapter
```

Safety cases in `data/eval_cases.jsonl` include unsupported / destructive phrasing that must map to `unsupported`.

## Few-shot × decode constraint (strict validation)

`training/ablation.py` → `artifacts/ablation_constraints.csv`, `artifacts/ablation_{dev,fresh,test}.json`.
Every row uses **strict** validation: unknown parameter keys are rejected, not dropped.

| Few-shot | Constraint | Dev (40) | Fresh (24) | Test (60) | Fallback (test) | Median latency (test) |
| :---: | --- | ---: | ---: | ---: | ---: | ---: |
| | none | 65.0% | 75.0% | 46.7% | 20.0% | 404 ms |
| | intent | 65.0% | 75.0% | 46.7% | 20.0% | 381 ms |
| | json | 77.5% | 87.5% | 58.3% | 0% | 240 ms |
| ✓ | none | 87.5% | 87.5% | 80.0% | 6.7% | 332 ms |
| ✓ | intent | 87.5% | 87.5% | 80.0% | 6.7% | 344 ms |
| ✓ | **json (default)** | **92.5%** | **91.7%** | **85.0%** | **0%** | 323 ms |

What this shows:

- **Few-shot examples are the largest gain** (dev 65.0 → 87.5%, test 46.7 → 80.0% with free decoding).
- **Strict validation exposes stray keys.** The model often adds parameters such as
  `git_status {"clean": …}` or `git_diff_summary {"summary": …}`. Rejected outputs fall back
  to the heuristic (5–7% of requests with examples, 15–20% without).
- **The JSON grammar removes them.** Decoding constrained to the decision schema can't produce
  stray keys: fallbacks drop to 0% on every set and exact accuracy rises 5 points on dev and
  test, at no latency cost (decode 12.8–13.9 ms/token in every mode).
- **The intent mask alone changes nothing**: every failure it could prevent is a parameter failure.
- **Grammars should constrain structure, not allowlisted values.** Making `open_app.app_name`
  an enum turned "open calculator" into Terminal; allowlists stay in the executor, which can refuse.

Caveat: one case is 2.5 / 4.2 / 1.7 points on dev / fresh / test, so single differences are
indicative, not significant; the ranking is the same on all three sets.

## Quantized KV and rotating windows

`training/bench_kv_variants.py` → `artifacts/kv_variants.json`.

| Variant | Cached prefix | Exact (dev + fresh, 64) | Same decisions as 16-bit | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 16-bit prefix (default) | 146.8 MB | 92.2% | — | 70 ms |
| 8-bit prefix (`model.kv_bits = 8`) | 78.0 MB | 92.2% | 59 / 64 | 72 ms |
| 4-bit prefix | 41.3 MB | 6.2% | 1 / 64 | 77 ms |
| No cache, full window | — | 92.2% | — | 1,868 ms |
| No cache, `max_kv_size` 1,024 / 512 / 256 | — | 25.0% / 15.6% / 9.4% | — | ≈2.1 s |

At 1.1k–8k tokens, 8-bit KV keeps the last-position logits within 1.1 of 16-bit (argmax and top-5
unchanged, KL < 1e-8). Its size follows `(bits/8 + 4/64) / 2` of 16-bit exactly. It gives no
speed-up up to 8k tokens, because decoding is bound by reading the 0.94 GB of weights. 4-bit KV
breaks Qwen3 even at 1.1k tokens (KL 25.8, argmax changed). A rotating window shorter than the
1.1k-token prompt drops most of the instructions. Both stay off by default. Of the five changed
8-bit decisions, three are free-text refusal reasons and two are intent flips, one each way.

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

## LoRA adapter vs base

`training/compare_lora.py` → `artifacts/lora_compare.json` (run `20261102T070331Z`: 700 steps,
8 layers, lr 1e-4, seq 1536, masked prompt; val loss 2.47 → 0.007; 78 min on the M4).

| Model | Constraint | Dev (40) | Fresh (24) | Test (60) | All 124 | Fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | json | 92.5% | 91.7% | 85.0% | 88.7% | 0% |
| Base | none | 87.5% | 87.5% | 80.0% | 83.9% | 6.7% (test) |
| LoRA | json | 100% | 100% | 93.3% | 96.8% | 0% |
| LoRA | none | 100% | 100% | 93.3% | 96.8% | 0% |

Paired exact McNemar (adapter vs base, all 124): json 12 fixed / 2 broken, p = 0.013;
none 18 / 2, p = 0.0004. Per set only test/none is significant alone (p = 0.039).
Remaining adapter misses (test): t14 `*.yaml` for "yml", t18 dockerfiles → recall_preference,
t36 preference value wording, t53 "commit everything" → git_status.



### Rank sweep (`training/lora_sweep.py`, `training/lora_subspace.py`; whitepaper Figs. 14–15):

| Rank r | Trainable params | log10 | Dev | Fresh | Test | All 124 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| — (base) | 0 | — | 92.5% | 91.7% | 85.0% | 88.7% |
| 1 | 311,296 | 5.49 | 95.0% | 100% | 95.0% | 96.0% |
| 4 | 1,245,184 | 6.10 | 100% | 100% | 93.3% | 96.8% |
| 8 | 2,490,368 | 6.40 | 100% | 100% | 93.3% | 96.8% |
| 64 | 19,922,944 | 7.30 | 90.0% | 87.5% | 81.7% | 85.5% |

A rank-1 update is already enough. r = 64 is confounded: mlx-lm keeps the multiplier fixed
(α/r = 20) instead of α, so the big adapter takes larger steps, and it was not retuned. Its update
is still effectively low-rank: the top 8 of 64 directions hold 70–87% of ‖ΔW‖², against 18% for a
random matrix. Unlike Hu et al., the A subspaces of the r = 8 and r = 64 runs overlap no better
than chance, because A stays close to its random initialisation in a 700-step run; the output
directions of ΔW overlap 20–60× more than chance.

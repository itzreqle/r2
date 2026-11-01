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

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

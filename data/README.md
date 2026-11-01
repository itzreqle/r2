# Data

Chat-style JSONL for MLX-LM and evaluation fixtures.

| File | Role |
| ---- | ---- |
| `train.jsonl` | Training examples |
| `valid.jsonl` | Validation during LoRA |
| `test.jsonl` | Held-out MLX-LM test split |
| `eval_cases.jsonl` | Intent evaluation (utterance → expected intent) |

Regenerate seed files:

```bash
uv run python training/prepare_data.py
uv run python training/validate_data.py
```

Do not commit private conversation logs or raw audio.

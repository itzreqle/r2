# Training workbench

Fine-tune a domain adapter with **MLX-LM LoRA/QLoRA** after the baseline text assistant works.

## Rules

1. Do not train until seed data is validated and baseline evaluation exists.
2. Never overwrite the base checkpoint; adapters go under `artifacts/runs/<run_id>/`.
3. Treat toy datasets as pipeline smoke tests only.
4. Inspect `mlx_lm.lora --help` before changing flags.

## Quick start (after model download approval)

```bash
uv sync
source .venv/bin/activate
uv run python training/validate_data.py --data-dir data
uv run python training/train.py --dry-run   # prints planned command
# Real train (downloads/uses cached Qwen3-1.7B-4bit):
uv run python training/train.py --iters 100 --yes
uv run python training/evaluate.py --adapter artifacts/runs/latest/adapters
uv run python training/compare_models.py
```

Conservative 16 GB defaults: `batch-size 1`, `num-layers 8`, `max-seq-length 1024`, `grad-checkpoint`.

# Training workbench

LoRA (QLoRA) fine-tuning of the routing decision with **MLX-LM**, on the 4-bit base model.

## Rules

1. The base checkpoint is never modified; adapters go under `artifacts/runs/<run_id>/adapters`.
2. Training uses only a model already in the local Hugging Face cache (`HF_HUB_OFFLINE=1` is
   forced; nothing is downloaded) and runs only with `--yes`.
3. Training data is synthetic and decontaminated: any utterance with word-Jaccard >= 0.6 to an
   evaluation case or a few-shot example is dropped.
4. The test set (`data/eval_test.jsonl`) was written before any training data.

## Pipeline

```bash
uv run python training/synth_data.py                  # data/synth/{train,valid}.jsonl + manifest
uv run python training/validate_data.py --data-dir data/synth
uv run python training/train.py --dry-run             # print the plan
HF_HUB_OFFLINE=1 uv run python training/train.py --data data/synth --iters 700 --yes
HF_HUB_OFFLINE=1 uv run python training/compare_lora.py   # base vs adapter, dev/fresh/test
```

Each example is the full R2 chat prompt plus the gold decision; `--mask-prompt` puts the loss on
the decision only. 16 GB defaults: `batch-size 1`, `num-layers 8`, `max-seq-length 1536` (the
prompt is ~1.15k tokens), `learning-rate 1e-4`, `grad-checkpoint`. Logs and `run_meta.json` go
next to the adapter; `artifacts/runs/latest` points at the last successful run.

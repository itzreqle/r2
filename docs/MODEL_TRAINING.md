# Model training

Base checkpoint: `mlx-community/Qwen3-1.7B-4bit` (Apache-2.0 via Qwen3). Adapter: LoRA on
the last 8 layers, trained with `mlx_lm.lora` (QLoRA, the base stays 4-bit).

## Procedure

1. Baseline first: `training/evaluate.py` on dev, fresh and test (`docs/EVALUATION.md`).
2. Generate data: `uv run python training/synth_data.py` (700 train / 70 valid, 11 intents,
   decontaminated against every eval set and the few-shot examples, see `manifest.json`).
3. Validate: `uv run python training/validate_data.py --data-dir data/synth`.
4. Dry-run: `uv run python training/train.py --dry-run`.
5. Train only with `--yes`. It refuses if the model is not cached; it never downloads.
6. Compare: `uv run python training/compare_lora.py` (exact accuracy with Wilson intervals,
   exact McNemar on paired cases) → `artifacts/lora_compare.json`.
7. Use an adapter with `model.adapter_path`; its content hash is part of the KV fingerprint.

## 16 GB defaults

- `--batch-size 1`, `--num-layers 8`, `--learning-rate 1e-4`, `--iters 700`
- `--max-seq-length 1536` (system prompt ~1.15k tokens)
- `--mask-prompt` (loss on the decision only), `--grad-checkpoint`
- Do not co-run Whisper/TTS/inference with training.

Inspect live flags: `uv run python -m mlx_lm.lora --help`

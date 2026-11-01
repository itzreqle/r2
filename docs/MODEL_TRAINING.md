# Model training

Preferred checkpoint: `mlx-community/Qwen3-1.7B-4bit` (Apache-2.0 via Qwen3).

## Procedure

1. Establish fake-backend + eval baseline.
2. Prepare JSONL: `uv run python training/prepare_data.py`
3. Validate: `uv run python training/validate_data.py`
4. Dry-run: `uv run python training/train.py --dry-run`
5. Train only with `--yes` (may download ~1 GB).
6. Evaluate adapter vs baseline; compare with `compare_models.py`.
7. Keep adapter selectable; never overwrite the base checkpoint.

## Conservative 16 GB defaults

- `--batch-size 1`
- `--num-layers 8`
- `--max-seq-length 1024`
- `--grad-checkpoint`
- Do not co-run Whisper/TTS/training with inference.

Inspect live flags: `uv run python -m mlx_lm.lora --help`

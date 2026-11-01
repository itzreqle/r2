# Notebooks

Interactive, comment-driven walkthrough. **CLIs and `src/narrow_ai/` remain the real usage path.**

| Notebook | Topic | Phase |
| -------- | ----- | ----- |
| [00_env_check.ipynb](00_env_check.ipynb) | Python 3.12, MLX device, versions | 1 |
| [01_intent_schemas.ipynb](01_intent_schemas.ipynb) | Schema validation & permissions | 1 |
| [02_tool_router_smoke.ipynb](02_tool_router_smoke.ipynb) | Fake backend → tools | 1 |
| [03_prompt_and_parser.ipynb](03_prompt_and_parser.ipynb) | Prompts & structured output | 2 |
| [04_mlx_inference_baseline.ipynb](04_mlx_inference_baseline.ipynb) | Load Qwen3 (explicit) | 2 |
| [05_dataset_eda.ipynb](05_dataset_eda.ipynb) | Training JSONL EDA | 3 |
| [06_lora_train_smoke.ipynb](06_lora_train_smoke.ipynb) | LoRA dry-run / train gate | 3 |
| [07_eval_compare.ipynb](07_eval_compare.ipynb) | Baseline vs adapter | 3 |
| [08_speech_loop.ipynb](08_speech_loop.ipynb) | STT/TTS abstractions | 4 |

## Kernel

```bash
cd /Users/siavash/Projects/articles/narrow-ai-with-kv
uv sync
uv run python -m ipykernel install --user --name=narrow-ai --display-name="Python (narrow-ai)"
uv run jupyter lab
```

Select **Python (narrow-ai)** in Jupyter / Cursor.

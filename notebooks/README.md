# Notebooks

Interactive walkthroughs of R2. They call the same code as the app and the `r2` command
(`src/narrow_ai/`), so what you see here is what R2 actually does.

| Notebook | What you do | Needs the model |
| --- | --- | --- |
| [00_env_check](00_env_check.ipynb) | Check Python 3.12, MLX device, package versions, cached models | no |
| [01_r2_playground](01_r2_playground.ipynb) | Chat with R2 through widgets; see intent, parameters, latency, cached tokens | yes |
| [02_kv_cache_lab](02_kv_cache_lab.ipynb) | Compare cache modes `none` / `prefix` / `persistent`; TTFT plots; KV memory maths | yes |
| [03_structured_output](03_structured_output.ipynb) | Response prefill and the intent-name token mask, step by step | yes |
| [04_eval](04_eval.ipynb) | Dev and fresh test sets, confusion matrix, few-shot × constrained ablation | yes |
| [05_voice_loop](05_voice_loop.ipynb) | `say` → Whisper → R2 round-trip with timings (no microphone); optional push-to-talk widget | yes |

"Needs the model" means Qwen3-1.7B-4bit (and Whisper for 05) must be in the Hugging
Face cache. They are downloaded the first time R2 runs; see the root README.

`archive/` keeps the original phase-1 stubs (schemas, router smoke test, LoRA dry run) for
reference. They are not maintained.

## Kernel

```bash
cd narrow-ai-with-kv
uv sync
uv run python -m ipykernel install --user --name=narrow-ai --display-name="Python (narrow-ai)"
uv run jupyter lab
```

Select **Python (narrow-ai)**, or the project's `.venv` in VS Code / VSCodium. The notebooks
work whether they are run from `notebooks/` or the repo root.

VSCodium can't see any Python environment until its missing `pet` finder is restored: run
`scripts/fix_vscodium_pet.sh`, then *Developer: Reload Window*.

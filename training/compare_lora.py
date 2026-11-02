#!/usr/bin/env python3
"""Base model vs LoRA adapter on the routing sets (cached models only, no tools run).

    HF_HUB_OFFLINE=1 uv run python training/compare_lora.py --adapter artifacts/runs/latest/adapters

For each of {base, adapter} × {JSON grammar (R2 default), no constraint} it routes the dev
(40), fresh (24) and test (60) sets with the normal R2 prompt. It reports exact accuracy
with Wilson 95% intervals, an exact McNemar test on the paired per-case outcomes
(adapter vs base, same constraint), fallback rate and TTFT. Writes artifacts/lora_compare.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.multistep import wilson  # noqa: E402
from narrow_ai.evaluation.runner import EvaluationRunner  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

SPLITS = {"dev": "eval_heldout.jsonl", "fresh": "eval_fresh.jsonl", "test": "eval_test.jsonl"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b = only base right, c = only adapter right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path,
                    default=ROOT / "artifacts" / "runs" / "latest" / "adapters")
    ap.add_argument("--constraints", default="json,none")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "lora_compare.json")
    args = ap.parse_args()
    if not (args.adapter / "adapters.safetensors").is_file():
        print(f"no adapter at {args.adapter}")
        return 2

    cases = {s: load_eval_cases(ROOT / "data" / f) for s, f in SPLITS.items()}
    results: dict[str, dict] = {}
    for model in ("base", "adapter"):
        settings = load_settings()
        settings.model.backend = "mlx"
        settings.model.cache_mode = "prefix"  # RAM only; nothing written to artifacts/kv
        settings.model.adapter_path = str(args.adapter.resolve()) if model == "adapter" else ""
        backend = create_backend(settings)
        for constraint in args.constraints.split(","):
            settings.model.constraint = constraint
            runner = EvaluationRunner(settings, backend)
            backend.warmup(runner.engine.system_prompt)
            runner.engine.decide("what can you do")
            key = f"{model}/{constraint}"
            results[key] = {}
            for split, cs in cases.items():
                rep = runner.run_cases(cs)
                ok = [r.params_ok for r in rep.cases]
                results[key][split] = {
                    "n": len(ok),
                    "exact": sum(ok) / len(ok),
                    "exact_ci95": [round(x, 3) for x in wilson(sum(ok), len(ok))],
                    "intent": sum(r.intent_ok for r in rep.cases) / len(ok),
                    "fallback": rep.fallback_rate,
                    "ttft_ms_median": statistics.median(
                        r.metrics.get("ttft_ms", 0) for r in rep.cases[1:]),
                    "ok": ok,
                    "cases": [{"id": r.id, "utterance": r.utterance, "expected": r.expected,
                               "predicted": r.predicted, "parameters": r.parameters,
                               "ok": r.params_ok, "source": r.source} for r in rep.cases],
                }
                print(f"{key:14} {split:5} exact {results[key][split]['exact']:.3f} "
                      f"fallback {rep.fallback_rate:.3f}", flush=True)
        backend.close()
        del backend
        gc.collect()

    paired = {}
    for constraint in args.constraints.split(","):
        for split in SPLITS:
            a = results[f"base/{constraint}"][split]["ok"]
            b = results[f"adapter/{constraint}"][split]["ok"]
            only_base = sum(x and not y for x, y in zip(a, b, strict=True))
            only_adapter = sum(y and not x for x, y in zip(a, b, strict=True))
            paired[f"{constraint}/{split}"] = {
                "only_base_right": only_base, "only_adapter_right": only_adapter,
                "mcnemar_p": round(mcnemar_exact(only_base, only_adapter), 4),
            }

    print("\n| Model | Constraint | Dev (40) | Fresh (24) | Test (60) | Fallback (test) | TTFT |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for key, r in results.items():
        m, c = key.split("/")
        print(f"| {m} | {c} | {r['dev']['exact']:.1%} | {r['fresh']['exact']:.1%} | "
              f"{r['test']['exact']:.1%} [{r['test']['exact_ci95'][0]:.0%}–"
              f"{r['test']['exact_ci95'][1]:.0%}] | {r['test']['fallback']:.1%} | "
              f"{r['test']['ttft_ms_median']:.0f} ms |")
    print("\nPaired (adapter vs base):", json.dumps(paired, indent=1))
    meta_path = args.adapter.parent / "run_meta.json"
    args.out.write_text(json.dumps({
        "adapter": str(args.adapter), "run_meta": json.loads(meta_path.read_text())
        if meta_path.exists() else None, "results": results, "paired": paired,
    }, indent=2, default=str))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

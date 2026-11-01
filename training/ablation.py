#!/usr/bin/env python3
"""Ablation: few-shot x decode-time constraint (none | intent | json) on real MLX.

One model load; every configuration runs over the dev set and the fresh test set.
Writes artifacts/ablation_constraints.csv (dev set; the original 2x2 artifacts/ablation.csv
used by the paper and README charts is left untouched)
and artifacts/ablation_<split>.json with per-config details. Uses the locally
cached model only (set HF_HUB_OFFLINE=1); nothing is downloaded or trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.runner import EvaluationRunner  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

SPLITS = {"dev": ROOT / "data" / "eval_heldout.jsonl", "fresh": ROOT / "data" / "eval_fresh.jsonl"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", default="none,intent,json")
    ap.add_argument("--few-shot", default="true,false")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args()

    settings = load_settings()
    settings.model.backend = "mlx"
    backend = create_backend(settings)
    rows: dict[str, list[dict]] = {k: [] for k in SPLITS}
    for few in [v.strip() == "true" for v in args.few_shot.split(",")]:
        for mode in [m.strip() for m in args.constraints.split(",")]:
            settings.model.few_shot = few
            settings.model.constraint = mode
            runner = EvaluationRunner(settings, backend)
            for split, path in SPLITS.items():
                rep = runner.run_cases(load_eval_cases(path))
                steps = [c.metrics.get("generated_tokens", 0) for c in rep.cases]
                gen_ms = [
                    c.metrics.get("total_ms", 0.0) - c.metrics.get("ttft_ms", 0.0)
                    for c in rep.cases
                ]
                row = {
                    "few_shot": few,
                    "constraint": mode,
                    "constrained": mode != "none",
                    "intent_acc": round(rep.metrics.accuracy, 3),
                    "param_acc": round(rep.param_accuracy, 3),
                    "malformed": round(rep.metrics.malformed_rate, 3),
                    "fallback": round(rep.fallback_rate, 3),
                    "latency_ms": round(rep.latency_ms_mean),
                    "ttft_ms": round(rep.ttft_ms_mean, 1),
                    "decode_ms_per_token": round(
                        sum(gen_ms) / max(1, sum(s - 1 for s in steps if s > 1)), 2
                    ),
                    "n": rep.metrics.n,
                    "misses": [
                        {"id": c.id, "utterance": c.utterance, "expected": c.expected,
                         "got": c.predicted, "params": c.parameters, "source": c.source}
                        for c in rep.cases if not (c.intent_ok and c.params_ok)
                    ],
                }
                rows[split].append(row)
                print(
                    f"{split:5} few_shot={few!s:5} constraint={mode:6} "
                    f"intent={row['intent_acc']:.3f} params={row['param_acc']:.3f} "
                    f"fallback={row['fallback']:.3f} latency={row['latency_ms']}ms "
                    f"ttft={row['ttft_ms']}ms decode={row['decode_ms_per_token']}ms/tok",
                    flush=True,
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in rows.items():
        (args.out_dir / f"ablation_{split}.json").write_text(json.dumps(data, indent=2))
    cols = ["few_shot", "constrained", "constraint", "intent_acc", "param_acc", "malformed",
            "fallback", "latency_ms"]
    with (args.out_dir / "ablation_constraints.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows["dev"], key=lambda r: (r["few_shot"], r["constraint"])))
    med = statistics.median(r["decode_ms_per_token"] for r in rows["dev"])
    print("median decode ms/token:", med)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare multi-step planners on data/eval_multistep.jsonl (plans only, nothing runs).

    uv run python training/eval_multistep.py                       # fake backend, offline
    HF_HUB_OFFLINE=1 uv run python training/eval_multistep.py --backend mlx

With the MLX backend this uses the locally cached model only; nothing is downloaded.
Writes artifacts/multistep_eval_<backend>.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.agent.planner import ClausePlanner, HeuristicPlanner, ModelPlanner  # noqa: E402
from narrow_ai.assistant.orchestrator import IntentEngine  # noqa: E402
from narrow_ai.evaluation.multistep import load_cases, report_dict, run_planner  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake", choices=["fake", "mlx"])
    ap.add_argument("--planners", default="")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "eval_multistep.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    settings.model.backend = args.backend
    backend = create_backend(settings)
    registry = default_registry()  # what the default config offers
    engine = IntentEngine(settings, backend, registry)
    if args.backend == "mlx":
        backend.warmup(engine.system_prompt)
    default = "heuristic,clause,clause+" + (",model" if args.backend == "mlx" else "")
    names = [p.strip() for p in (args.planners or default).split(",") if p.strip()]
    planners = {
        "heuristic": HeuristicPlanner(),
        "clause": ClausePlanner(engine, settings.agent.max_steps),
        "clause+": ClausePlanner(engine, settings.agent.max_steps,
                                 split_and=True, question_guard=True),
        "model": ModelPlanner(settings, backend, registry) if args.backend == "mlx" else None,
    }
    cases = load_cases(args.data)
    reports = []
    for name in names:
        planner = planners[name]
        if planner is None:
            print(f"skipping {name}: needs --backend mlx")
            continue
        if name == "model":
            backend.warmup(planner.system_prompt)

        def show(r, name=name):
            if args.verbose and not r.plan_ok:
                print(f"  [{name}] {r.id} {r.utterance!r}\n     expected {r.expected}\n"
                      f"     got      {r.got} {r.error}")

        rep = run_planner(name, planner, registry, cases, progress=show)
        reports.append(rep)

    print(f"\n{len(cases)} requests, backend={args.backend}\n")
    print("| Planner | Plan exact | 95% CI | Intent seq. | Step count | Reject recall "
          "| False reject | Latency p50 |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in reports:
        lo, hi = r.plan_ci95
        print(f"| {r.planner} | {r.plan_accuracy:.1%} | {lo:.1%}–{hi:.1%} | "
              f"{r.intent_sequence_accuracy:.1%} | {r.step_count_accuracy:.1%} | "
              f"{r.reject_recall:.1%} | {r.false_reject_rate:.1%} | {r.latency_ms_p50:.0f} ms |")
    cats = sorted({c for r in reports for c in r.by_category})
    print("\n| Category | n | " + " | ".join(r.planner for r in reports) + " |")
    print("|---|---:|" + "---:|" * len(reports))
    for c in cats:
        n = reports[0].by_category.get(c, {}).get("n", 0)
        vals = " | ".join(
            f"{r.by_category.get(c, {}).get('plan_accuracy', 0):.0%}" for r in reports
        )
        print(f"| {c} | {n:.0f} | {vals} |")

    out = args.out or ROOT / "artifacts" / f"multistep_eval_{args.backend}.json"
    out.write_text(json.dumps(
        {"backend": args.backend, "data": args.data.resolve().name,
         "constraint": settings.model.constraint_mode(),
         "reports": [report_dict(r) for r in reports]}, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare two evaluation JSON reports (baseline vs adapter)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/eval_baseline.json"))
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/eval_adapter.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/compare_report.json"))
    args = parser.parse_args()

    if not args.baseline.is_file() or not args.adapter.is_file():
        print("Missing baseline and/or adapter reports. Run training/evaluate.py twice.")
        print(f"  expected: {args.baseline} and {args.adapter}")
        return 1

    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    adap = json.loads(args.adapter.read_text(encoding="utf-8"))
    report = {
        "baseline_accuracy": base.get("accuracy"),
        "adapter_accuracy": adap.get("accuracy"),
        "delta_accuracy": (adap.get("accuracy") or 0) - (base.get("accuracy") or 0),
        "baseline_malformed_rate": base.get("malformed_rate"),
        "adapter_malformed_rate": adap.get("malformed_rate"),
        "improved": (adap.get("accuracy") or 0) > (base.get("accuracy") or 0),
        "note": "Adapter must not become default solely because training finished.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

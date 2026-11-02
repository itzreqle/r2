#!/usr/bin/env python3
"""LoRA rank sweep: accuracy vs number of trainable parameters (cached model only).

    HF_HUB_OFFLINE=1 uv run python training/lora_sweep.py --ranks 1,4,64 --yes

Trains one adapter per rank (same data, layers, steps and seed as the main run) under
artifacts/runs/sweep-r<rank>/, then routes dev + fresh + test (124 requests) with the
default JSON grammar. The existing main run (artifacts/runs/latest) is included as r = 8.
Writes artifacts/lora_sweep.json. Nothing runs without --yes.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.evaluation.dataset import load_eval_cases  # noqa: E402
from narrow_ai.evaluation.runner import EvaluationRunner  # noqa: E402
from narrow_ai.models.model_registry import create_backend  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

SPLITS = {"dev": "eval_heldout.jsonl", "fresh": "eval_fresh.jsonl", "test": "eval_test.jsonl"}
LAYERS = 8
# per-layer sum of (d + k) over q, k, v, o, gate, up, down for Qwen3-1.7B
SUM_DK = 4096 * 2 + 3072 * 2 + 8192 * 3  # = 38,912


def n_params(rank: int, layers: int = LAYERS) -> int:
    return layers * rank * SUM_DK


def val_losses(log: Path) -> list[list[float]]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    pat = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)", re.M)
    return [[int(m[1]), float(m[2])] for m in pat.finditer(text)]


def evaluate(adapter: Path) -> dict:
    settings = load_settings()
    settings.model.backend = "mlx"
    settings.model.cache_mode = "prefix"
    settings.model.constraint = "json"
    settings.model.adapter_path = str(adapter.resolve())
    backend = create_backend(settings)
    runner = EvaluationRunner(settings, backend)
    backend.warmup(runner.engine.system_prompt)
    runner.engine.decide("what can you do")
    out = {}
    for split, f in SPLITS.items():
        rep = runner.run_cases(load_eval_cases(ROOT / "data" / f))
        ok = [r.params_ok for r in rep.cases]
        out[split] = {"n": len(ok), "exact": sum(ok) / len(ok), "fallback": rep.fallback_rate}
        print(f"  {split:5} exact {out[split]['exact']:.3f}", flush=True)
    backend.close()
    del backend
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="1,4,64")
    ap.add_argument("--iters", type=int, default=700)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "lora_sweep.json")
    args = ap.parse_args()
    if not args.yes:
        print("Dry run: re-run with --yes to train (about 55 min per rank on an M4).")
        return 0
    result = json.loads(args.out.read_text()) if args.out.exists() else {"runs": {}}
    main_run = ROOT / "artifacts" / "runs" / "latest"
    cmp = ROOT / "artifacts" / "lora_compare.json"
    if cmp.exists() and "8" not in result["runs"]:
        res = json.loads(cmp.read_text())["results"]
        result["base"] = {s: res["base/json"][s] | {"ok": None, "cases": None} for s in SPLITS}
        result["runs"]["8"] = {
            "rank": 8, "params": n_params(8), "val_loss": val_losses(main_run / "train.log"),
            "eval": {s: {k: res["adapter/json"][s][k] for k in ("n", "exact", "fallback")}
                     for s in SPLITS}}
    for rank in (int(r) for r in args.ranks.split(",")):
        if str(rank) in result["runs"]:
            continue
        run = ROOT / "artifacts" / "runs" / f"sweep-r{rank}"
        print(f"== rank {rank}: {n_params(rank):,} params", flush=True)
        code = subprocess.call([
            sys.executable, str(ROOT / "training" / "train.py"), "--yes", "--rank", str(rank),
            "--iters", str(args.iters), "--steps-per-eval", str(args.iters),
            "--adapter-path", str(run / "adapters")], stdout=subprocess.DEVNULL)
        if code != 0:
            print(f"training failed for rank {rank} (exit {code})")
            return code
        result["runs"][str(rank)] = {"rank": rank, "params": n_params(rank),
                                     "val_loss": val_losses(run / "train.log"),
                                     "eval": evaluate(run / "adapters")}
        args.out.write_text(json.dumps(result, indent=1))
    for r in sorted(result["runs"].values(), key=lambda r: r["params"]):
        ok = sum(r["eval"][s]["exact"] * r["eval"][s]["n"] for s in SPLITS)
        n = sum(r["eval"][s]["n"] for s in SPLITS)
        print(f"r={r['rank']:3} params={r['params']:>10,} val={r['val_loss'][-1][1]:.4f} "
              f"exact={ok / n:.3f}")
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""LoRA training wrapper around ``mlx_lm.lora`` (QLoRA on the 4-bit base model).

    uv run python training/train.py --dry-run                      # print the plan
    uv run python training/train.py --data data/synth --iters 700 --yes

Rules (see training/README.md): the base checkpoint is never modified; adapters go to
``artifacts/runs/<run_id>/adapters``; training only uses a model that is already in the
local Hugging Face cache (``HF_HUB_OFFLINE=1`` is forced, so nothing is downloaded); and
nothing runs without ``--yes``.

The loss covers only the assistant's decision (``--mask-prompt``), not the ~1.1k-token
system prompt, which is identical in every example.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synth")
    parser.add_argument("--iters", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=1536)  # prompt is ~1.15k tokens
    parser.add_argument("--steps-per-eval", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=-1, help="-1 = whole valid set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm real training")
    args = parser.parse_args()

    from narrow_ai.models.hub import local_or_repo

    source = local_or_repo(args.model)
    if not Path(source).is_dir():
        print(f"{args.model} is not in the local cache ({source}). Training never downloads;"
              " fetch the model explicitly first.")
        return 2

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.adapter_path or ROOT / "artifacts" / "runs" / run_id / "adapters"
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", source,
        "--train",
        "--data", str(args.data),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--learning-rate", str(args.learning_rate),
        "--max-seq-length", str(args.max_seq_length),
        "--steps-per-eval", str(args.steps_per_eval),
        "--val-batches", str(args.val_batches),
        "--seed", str(args.seed),
        "--mask-prompt",
        "--grad-checkpoint",
        "--adapter-path", str(out),
    ]
    meta = {
        "run_id": run_id, "model": args.model, "model_source": source, "data": str(args.data),
        "iters": args.iters, "batch_size": args.batch_size, "num_layers": args.num_layers,
        "learning_rate": args.learning_rate, "max_seq_length": args.max_seq_length,
        "seed": args.seed, "mask_prompt": True, "command": cmd,
    }
    manifest = args.data / "manifest.json"
    if manifest.is_file():
        meta["data_manifest"] = json.loads(manifest.read_text())
    print(json.dumps({k: v for k, v in meta.items() if k != "data_manifest"}, indent=2))

    if args.dry_run or not args.yes:
        print("Dry run. Re-run with --yes to train (cached model only, no download).")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log = out.parent / "train.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
    print(f"log: {log}")
    if proc.returncode == 0:
        latest = ROOT / "artifacts" / "runs" / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out.parent.resolve())
    else:
        print(f"training failed with exit code {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

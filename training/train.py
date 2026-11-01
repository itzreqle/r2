#!/usr/bin/env python3
"""LoRA/QLoRA training wrapper around mlx_lm.lora."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="Confirm real training (may download model)"
    )
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.adapter_path or Path("artifacts") / "runs" / run_id / "adapters"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(args.data),
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--max-seq-length",
        str(args.max_seq_length),
        "--grad-checkpoint",
        "--adapter-path",
        str(out),
    ]

    meta = {
        "run_id": run_id,
        "model": args.model,
        "command": cmd,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "max_seq_length": args.max_seq_length,
    }
    meta_path = out.parent / "run_meta.json"
    print(json.dumps(meta, indent=2))

    if args.dry_run or not args.yes:
        print("Dry run / confirmation required. Re-run with --yes to train.")
        print("Estimated: ~1 GB download if model not cached; keep other heavy jobs closed.")
        return 0

    if shutil.which("mlx_lm.lora") is None and True:
        # Prefer python -m for isolation
        pass

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Inspect help once for transparency
    subprocess.run([sys.executable, "-m", "mlx_lm.lora", "--help"], check=False)
    proc = subprocess.run(cmd, check=False)
    latest = Path("artifacts") / "runs" / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out.parent.resolve())
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

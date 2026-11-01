#!/usr/bin/env python3
"""Document / optionally fuse an adapter. Does not overwrite the base checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record adapter export metadata. Fusion is optional and explicit."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--out", type=Path, default=Path("artifacts/export_meta.json"))
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Attempt mlx_lm fuse (writes a NEW directory; never overwrites base)",
    )
    parser.add_argument("--fuse-out", type=Path, default=Path("artifacts/fused-model"))
    args = parser.parse_args()

    meta = {
        "base_model": args.model,
        "adapter": str(args.adapter),
        "fused": False,
        "fuse_out": None,
    }
    if args.fuse:
        import subprocess
        import sys

        args.fuse_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            args.model,
            "--adapter-path",
            str(args.adapter),
            "--save-path",
            str(args.fuse_out),
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        meta["fused"] = proc.returncode == 0
        meta["fuse_out"] = str(args.fuse_out)
        meta["returncode"] = proc.returncode

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if not args.fuse or meta.get("fused") else 1


if __name__ == "__main__":
    raise SystemExit(main())

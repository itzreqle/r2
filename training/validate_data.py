#!/usr/bin/env python3
"""Validate MLX-LM chat JSONL datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_ROLES = {"system", "user", "assistant"}


def validate_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    n = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{i} invalid json: {exc}")
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                errors.append(f"{path}:{i} missing messages")
                continue
            roles = {m.get("role") for m in msgs if isinstance(m, dict)}
            if not {"user", "assistant"} <= roles:
                errors.append(f"{path}:{i} need user+assistant roles")
            key = json.dumps(obj, sort_keys=True)
            if key in seen:
                errors.append(f"{path}:{i} duplicate example")
            seen.add(key)
            # assistant content should be JSON intent when possible
            for m in msgs:
                if m.get("role") == "assistant":
                    try:
                        json.loads(m["content"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        errors.append(f"{path}:{i} assistant content not JSON")
    return n, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    all_errors: list[str] = []
    counts: Counter[str] = Counter()
    for name in ("train", "valid", "test"):
        path = args.data_dir / f"{name}.jsonl"
        if not path.is_file():
            if name != "test":  # mlx-lm needs train + valid; test is optional
                all_errors.append(f"missing {path}")
            continue
        n, errors = validate_file(path)
        counts[name] = n
        all_errors.extend(errors)
    print(dict(counts))
    if all_errors:
        print("ERRORS:")
        for e in all_errors:
            print(" -", e)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

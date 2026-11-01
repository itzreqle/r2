#!/usr/bin/env python3
"""Prepare / expand seed chat JSONL for MLX-LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = (
    "You are a narrow macOS developer voice assistant. "
    "Output concise, valid JSON intents."
)

SEED = [
    ("Check the current Git status.", {"intent": "git_status", "parameters": {}, "requires_confirmation": False}),
    ("What's my git status?", {"intent": "git_status", "parameters": {}, "requires_confirmation": False}),
    ("Find Python files in the project.", {"intent": "search_files", "parameters": {"pattern": "*.py", "workspace": "active"}, "requires_confirmation": False}),
    ("Search for markdown files.", {"intent": "search_files", "parameters": {"pattern": "*.md", "workspace": "active"}, "requires_confirmation": False}),
    ("What can you do?", {"intent": "general_help", "parameters": {}, "requires_confirmation": False}),
    ("Tell me a joke.", {"intent": "unsupported", "parameters": {"reason": "Outside supported domain"}, "requires_confirmation": False}),
    ("Delete all files.", {"intent": "unsupported", "parameters": {"reason": "Destructive action not allowed"}, "requires_confirmation": False}),
    ("Run rm -rf /", {"intent": "unsupported", "parameters": {"reason": "Arbitrary shell not allowed"}, "requires_confirmation": False}),
    ("Remember preferred_editor=Cursor", {"intent": "remember_preference", "parameters": {"key": "preferred_editor", "value": "Cursor"}, "requires_confirmation": True}),
    ("Recall preferred_editor", {"intent": "recall_preference", "parameters": {"key": "preferred_editor"}, "requires_confirmation": False}),
    ("status", {"intent": "clarify_request", "parameters": {"question": "Do you mean git status?"}, "requires_confirmation": False}),
    ("Open my bank app and transfer money", {"intent": "unsupported", "parameters": {"reason": "Outside supported domain"}, "requires_confirmation": False}),
]


def to_record(user: str, assistant_obj: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [to_record(u, a) for u, a in SEED]
    # Simple split: last 2 -> test, next-to-last 2 -> valid, rest train
    train, valid, test = records[:-4], records[-4:-2], records[-2:]
    for name, rows in ("train", train), ("valid", valid), ("test", test):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")


if __name__ == "__main__":
    main()

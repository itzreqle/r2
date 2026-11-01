#!/usr/bin/env python3
"""Prepare / expand seed chat JSONL for MLX-LM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.assistant.orchestrator import system_prompt_for  # noqa: E402
from narrow_ai.intents.registry import default_registry  # noqa: E402
from narrow_ai.settings import load_settings  # noqa: E402

# Exactly the prompt R2 uses at runtime (no train/serve skew).
SYSTEM = system_prompt_for(load_settings(), default_registry())

SEED = [
    ("Check the current Git status.", {"intent": "git_status", "parameters": {}}),
    ("What's my git status?", {"intent": "git_status", "parameters": {}}),
    ("Find Python files in the project.", {"intent": "search_files", "parameters": {"pattern": "*.py", "workspace": "active"}}),
    ("Search for markdown files.", {"intent": "search_files", "parameters": {"pattern": "*.md", "workspace": "active"}}),
    ("What can you do?", {"intent": "general_help", "parameters": {}}),
    ("Tell me a joke.", {"intent": "unsupported", "parameters": {"reason": "Outside supported domain"}}),
    ("Delete all files.", {"intent": "unsupported", "parameters": {"reason": "Destructive action not allowed"}}),
    ("Run rm -rf /", {"intent": "unsupported", "parameters": {"reason": "Arbitrary shell not allowed"}}),
    ("Remember preferred_editor=Cursor", {"intent": "remember_preference", "parameters": {"key": "preferred_editor", "value": "Cursor"}}),
    ("Recall preferred_editor", {"intent": "recall_preference", "parameters": {"key": "preferred_editor"}}),
    ("status", {"intent": "clarify_request", "parameters": {"question": "Do you mean git status?"}}),
    ("Open my bank app and transfer money", {"intent": "unsupported", "parameters": {"reason": "Outside supported domain"}}),
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

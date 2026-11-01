"""Allowlisted test command runner (disabled by default)."""

from __future__ import annotations

import subprocess
from pathlib import Path


class TestToolError(RuntimeError):
    pass


def run_allowlisted_test(
    workspace: Path,
    command: list[str],
    *,
    timeout: int = 120,
    max_chars: int = 8000,
) -> str:
    if not command:
        raise TestToolError("empty command")
    # Never use shell=True
    try:
        proc = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TestToolError("test command timed out") from exc
    out = (proc.stdout or "") + (proc.stderr or "")
    return f"exit={proc.returncode}\n{out}"[:max_chars]

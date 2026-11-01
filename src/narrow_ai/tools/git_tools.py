"""Read-only Git helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitToolError(RuntimeError):
    pass


def _run_git(workspace: Path, args: list[str], timeout: int) -> str:
    cmd = ["git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitToolError("git executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitToolError("git command timed out") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "git failed").strip()
        raise GitToolError(err[:2000])
    return (proc.stdout or "").strip()


def git_status(workspace: Path, *, timeout: int = 30, max_chars: int = 8000) -> str:
    branch = _run_git(workspace, ["rev-parse", "--abbrev-ref", "HEAD"], timeout)
    status = _run_git(workspace, ["status", "--short", "--branch"], timeout)
    text = f"branch: {branch}\n{status}"
    return text[:max_chars]


def git_diff_stat(workspace: Path, *, timeout: int = 30, max_chars: int = 8000) -> str:
    diff = _run_git(workspace, ["diff", "--stat"], timeout)
    staged = _run_git(workspace, ["diff", "--cached", "--stat"], timeout)
    parts = []
    if diff:
        parts.append(f"unstaged:\n{diff}")
    if staged:
        parts.append(f"staged:\n{staged}")
    if not parts:
        return "No changes."
    return "\n\n".join(parts)[:max_chars]

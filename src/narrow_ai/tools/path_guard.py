"""Workspace path confinement."""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(PermissionError):
    pass


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    workspace = workspace.resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise PathEscapeError(f"Path escapes workspace: {relative}") from exc
    return candidate


def is_under_workspace(workspace: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
        return True
    except ValueError:
        return False

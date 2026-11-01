"""Workspace file search (read-only)."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from narrow_ai.tools.path_guard import PathEscapeError, resolve_in_workspace

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    ".build",  # SwiftPM
    ".swiftpm",
    "DerivedData",  # Xcode
    ".ipynb_checkpoints",
    ".mypy_cache",
}


def search_files(
    workspace: Path,
    pattern: str,
    *,
    max_results: int = 50,
) -> list[str]:
    workspace = workspace.resolve()
    matches: list[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        try:
            rel = path.relative_to(workspace).as_posix()
        except ValueError:
            continue
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
            if len(matches) >= max_results:
                break
    return matches


def read_workspace_file(workspace: Path, relative: str, *, max_chars: int = 8000) -> str:
    path = resolve_in_workspace(workspace, relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    # Refuse likely secrets
    name = path.name.lower()
    if (
        name.startswith(".env")
        or name in {"credentials.json", "id_rsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc"}
        or name.endswith((".pem", ".key", ".p12"))
    ):
        raise PathEscapeError(f"Refusing to read sensitive file: {relative}")
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]

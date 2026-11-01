"""Workspace file search and read (read-only, symlink- and race-aware)."""

from __future__ import annotations

import fnmatch
import os
import re
import stat
import sys
from pathlib import Path

from narrow_ai.tools.path_guard import PathEscapeError, is_under_workspace, resolve_in_workspace

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

_SENSITIVE_NAMES = {"credentials.json", "id_rsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc"}
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def is_sensitive_name(name: str) -> bool:
    name = name.lower()
    return (
        name.startswith(".env")
        or name in _SENSITIVE_NAMES
        or name.endswith(_SENSITIVE_SUFFIXES)
    )


def display_name(rel: str) -> str:
    """Filenames may contain newlines/control chars; never let them fake output lines."""
    return _CONTROL.sub("?", rel)


def search_files(
    workspace: Path,
    pattern: str,
    *,
    max_results: int = 50,
) -> list[str]:
    """Match file names under the workspace.

    Symlinked directories are never descended into, and symlinked files whose
    target lies outside the workspace are skipped, so search can't enumerate
    anything outside the workspace.
    """
    workspace = workspace.resolve()
    matches: list[str] = []
    for root, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        base = Path(root)
        for name in sorted(files):
            path = base / name
            if path.is_symlink() and not is_under_workspace(workspace, path):
                continue
            rel = path.relative_to(workspace).as_posix()
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                matches.append(rel)
                if len(matches) >= max_results:
                    return matches
    return matches


def _fd_path(fd: int) -> str | None:
    """Real path of an open file descriptor (macOS F_GETPATH), if available."""
    if sys.platform == "darwin":
        import fcntl

        try:
            buf = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
            return os.fsdecode(buf.split(b"\0", 1)[0])
        except OSError:
            return None
    proc = f"/proc/self/fd/{fd}"
    try:
        return os.readlink(proc)
    except OSError:
        return None


def read_workspace_file(workspace: Path, relative: str, *, max_chars: int = 8000) -> str:
    """Read a regular file inside the workspace.

    Defends against symlink escapes and the resolve→open race: the resolved path
    is opened with ``O_NOFOLLOW`` (``O_NONBLOCK`` so a FIFO can't hang us), then
    the *opened* descriptor's real path is re-checked against the workspace.
    """
    workspace = workspace.resolve()
    if is_sensitive_name(Path(relative).name):
        raise PathEscapeError(f"Refusing to read sensitive file: {relative}")
    path = resolve_in_workspace(workspace, relative)
    if is_sensitive_name(path.name):  # e.g. notes.txt -> .env
        raise PathEscapeError(f"Refusing to read sensitive file: {relative}")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise FileNotFoundError(relative) from None
    except OSError as exc:  # ELOOP: swapped for a symlink after resolve()
        raise PathEscapeError(f"Refusing to open {relative}: {exc.strerror}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            if stat.S_ISDIR(st.st_mode):
                raise FileNotFoundError(relative)
            raise PathEscapeError(f"Not a regular file: {relative}")
        real = _fd_path(fd)
        if real is not None:
            if not is_under_workspace(workspace, Path(real)) or is_sensitive_name(
                Path(real).name
            ):
                raise PathEscapeError(f"Path escapes workspace: {relative}")
        else:  # no fd→path API: at least make sure it's still the file we resolved
            now = os.stat(path, follow_symlinks=False)
            if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino):
                raise PathEscapeError(f"File changed while opening: {relative}")
        chunks: list[bytes] = []
        remaining = max_chars * 4  # UTF-8 worst case
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8", errors="replace")[:max_chars]

from __future__ import annotations

from pathlib import Path

import pytest

from narrow_ai.tools.path_guard import PathEscapeError, resolve_in_workspace


def test_resolve_ok(tmp_path: Path):
    p = resolve_in_workspace(tmp_path, "src/foo.py")
    assert p == (tmp_path / "src/foo.py").resolve()


def test_resolve_blocks_escape(tmp_path: Path):
    with pytest.raises(PathEscapeError):
        resolve_in_workspace(tmp_path, "../outside.txt")

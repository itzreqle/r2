from __future__ import annotations

from pathlib import Path

from narrow_ai.tools.file_tools import search_files


def test_search_py(tmp_path: Path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    hits = search_files(tmp_path, "*.py")
    assert hits == ["a.py"]

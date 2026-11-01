from __future__ import annotations

from pathlib import Path

from narrow_ai.tools.file_tools import search_files


def test_search_py(tmp_path: Path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    hits = search_files(tmp_path, "*.py")
    assert hits == ["a.py"]


def test_search_skips_build_outputs(tmp_path: Path):
    (tmp_path / "App.swift").write_text("x", encoding="utf-8")
    for junk in (".build/debug", ".swiftpm", "DerivedData/x"):
        (tmp_path / junk).mkdir(parents=True)
        (tmp_path / junk / "Cache.swift").write_text("y", encoding="utf-8")
    assert search_files(tmp_path, "*.swift") == ["App.swift"]


def test_language_words_become_extension_globs():
    from narrow_ai.intents.classifier import normalize_decision_data

    def pattern(p: str) -> str:
        data = {"intent": "search_files", "parameters": {"pattern": p}}
        return normalize_decision_data(data)["parameters"]["pattern"]

    assert pattern("swift") == "*.swift"
    assert pattern("*swift*") == "*.swift"
    assert pattern("Swift files") == "*.swift"
    assert pattern("*.swift") == "*.swift"
    assert pattern("config") == "*config*"
    assert pattern("*config*") == "*config*"
    assert pattern("src/*.py") == "src/*.py"

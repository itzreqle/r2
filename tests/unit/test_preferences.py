from __future__ import annotations

from pathlib import Path

import pytest

from narrow_ai.memory.preferences import PreferenceStore


def test_kv_roundtrip(tmp_path: Path):
    store = PreferenceStore(tmp_path / "prefs.db", max_preferences=5)
    store.set("editor", "Cursor")
    assert store.get("editor") == "Cursor"
    assert store.list_keys() == ["editor"]
    assert store.delete("editor")
    assert store.get("editor") is None


def test_max_preferences(tmp_path: Path):
    store = PreferenceStore(tmp_path / "prefs.db", max_preferences=2)
    store.set("a", "1")
    store.set("b", "2")
    with pytest.raises(RuntimeError):
        store.set("c", "3")

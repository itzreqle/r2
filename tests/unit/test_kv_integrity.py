"""KV cache integrity: content fingerprints, layer validation, eviction (no MLX needed)."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import pytest

from narrow_ai.models.fingerprint import (
    adapter_fingerprint,
    file_digest,
    snapshot_fingerprint,
)
from narrow_ai.models.kv_cache import TMP_SUFFIX, evict_cache_files, validate_layers

COMMIT = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"


def _blob(repo: Path, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    (repo / "blobs" / sha).write_bytes(data)
    return sha


def make_snapshot(tmp: Path, files: dict[str, bytes], commit: str = COMMIT) -> Path:
    """Mimic the HF cache: snapshots/<commit>/<file> -> ../../blobs/<content hash>."""
    repo = tmp / "models--org--model"
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        sha = _blob(repo, data)
        link = snap / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / ".." / "blobs" / sha)
    return snap


FILES = {
    "model.safetensors": b"weights-v1" * 100,
    "config.json": b'{"hidden_size": 8}',
    "tokenizer.json": b'{"vocab": {}}',
    "tokenizer_config.json": b'{"chat_template": "x"}',
    "README.md": b"docs",
}


# --------------------------------------------------------------------- fingerprints
def test_hf_snapshot_uses_revision_and_blob_hashes(tmp_path):
    snap = make_snapshot(tmp_path, FILES)
    fp = snapshot_fingerprint(snap)
    assert fp["revision"] == COMMIT
    assert all(v not in {"none", "unresolved"} for v in fp.values())
    expected = hashlib.sha256(FILES["model.safetensors"]).hexdigest()
    assert file_digest(snap / "model.safetensors") == f"blob:{expected}"


def test_same_content_gives_same_fingerprint(tmp_path):
    a = snapshot_fingerprint(make_snapshot(tmp_path / "a", FILES))
    b = snapshot_fingerprint(make_snapshot(tmp_path / "b", FILES))
    assert a == b


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        ("model.safetensors", "weights_sha"),
        ("tokenizer.json", "tokenizer_sha"),
        ("tokenizer_config.json", "tokenizer_sha"),
        ("config.json", "config_sha"),
    ],
)
def test_changed_file_changes_its_digest(tmp_path, changed, field):
    before = snapshot_fingerprint(make_snapshot(tmp_path / "a", FILES))
    after = snapshot_fingerprint(
        make_snapshot(tmp_path / "b", {**FILES, changed: FILES[changed] + b" "})
    )
    assert before[field] != after[field]
    others = {k for k in before if k not in {field, "revision"}}
    assert all(before[k] == after[k] for k in others)


def test_new_weights_under_same_repo_id_are_detected(tmp_path):
    """The case the old fingerprint (repo id only) missed: same id, new snapshot."""
    old = snapshot_fingerprint(make_snapshot(tmp_path, FILES, commit="a" * 40))
    new = snapshot_fingerprint(
        make_snapshot(tmp_path, {**FILES, "model.safetensors": b"weights-v2"}, commit="b" * 40)
    )
    assert old["revision"] != new["revision"] and old["weights_sha"] != new["weights_sha"]


def test_readme_does_not_affect_fingerprint(tmp_path):
    a = snapshot_fingerprint(make_snapshot(tmp_path / "a", FILES))
    b = snapshot_fingerprint(make_snapshot(tmp_path / "b", {**FILES, "README.md": b"new docs"}))
    assert a == b


def test_plain_local_dir_is_hashed(tmp_path):
    d = tmp_path / "my-model"
    d.mkdir()
    for name, data in FILES.items():
        (d / name).write_bytes(data)
    fp = snapshot_fingerprint(d)
    assert fp["revision"] == "local"
    (d / "model.safetensors").write_bytes(b"retrained")
    assert snapshot_fingerprint(d)["weights_sha"] != fp["weights_sha"]


def test_large_local_file_falls_back_to_size_and_mtime(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"x" * 100)
    d1 = file_digest(f, hash_limit=10)
    assert d1.startswith("stat:100:")
    os.utime(f, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
    assert file_digest(f, hash_limit=10) != d1


def test_unresolved_source_is_marked(tmp_path):
    fp = snapshot_fingerprint("org/not-downloaded")
    assert set(fp.values()) == {"unresolved"}


def test_adapter_fingerprint_tracks_content_not_path(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "adapters.safetensors").write_bytes(b"lora")
        (d / "adapter_config.json").write_text('{"rank": 8}')
        (d / "0000100_adapters.safetensors").write_bytes(b"old checkpoint")
    assert adapter_fingerprint(a) == adapter_fingerprint(b)  # path doesn't matter
    (b / "0000100_adapters.safetensors").write_bytes(b"other checkpoint")
    assert adapter_fingerprint(a) == adapter_fingerprint(b)  # unused checkpoints ignored
    (b / "adapters.safetensors").write_bytes(b"retrained")
    assert adapter_fingerprint(a) != adapter_fingerprint(b)
    assert adapter_fingerprint(None) == "none"
    assert adapter_fingerprint(tmp_path / "gone") == "missing"


# --------------------------------------------------------------------- layer validation
class KVCache:
    def __init__(self, offset: int, positions: int | None = None) -> None:
        self.offset = offset
        n = offset if positions is None else positions
        self.keys = type("Arr", (), {"shape": (1, 8, n, 64)})()


class RotatingKVCache(KVCache):
    pass


def test_validate_layers_accepts_consistent_cache():
    cache = [KVCache(100) for _ in range(4)]
    assert validate_layers(cache, ["KVCache"] * 4, 100) is None


@pytest.mark.parametrize(
    ("cache", "why"),
    [
        ([KVCache(100)] * 3, "layer count"),
        ([KVCache(100), KVCache(100), KVCache(99), KVCache(100)], "layer 2: offset"),
        ([KVCache(100)] * 3 + [KVCache(100, positions=50)], "layer 3: keys"),
        ([KVCache(100)] * 3 + [RotatingKVCache(100)], "layer 3: RotatingKVCache"),
    ],
)
def test_validate_layers_rejects_damage_in_any_layer(cache, why):
    """The old check only looked at cache[0].offset; every one of these passed it."""
    assert cache[0].offset == 100
    problem = validate_layers(cache, ["KVCache"] * 4, 100)
    assert problem is not None and why in problem


# --------------------------------------------------------------------- eviction
def _touch(path: Path, age_s: float) -> Path:
    path.write_bytes(b"x")
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def test_eviction_keeps_most_recent(tmp_path):
    files = [_touch(tmp_path / f"prefix-{i}.safetensors", age_s=i * 10) for i in range(6)]
    removed = evict_cache_files(tmp_path, 3)
    assert {p.name for p in removed} == {f.name for f in files[3:]}
    assert sorted(p.name for p in tmp_path.iterdir()) == [f.name for f in files[:3]]


def test_eviction_never_removes_files_in_use(tmp_path):
    files = [_touch(tmp_path / f"prefix-{i}.safetensors", age_s=i * 10) for i in range(4)]
    evict_cache_files(tmp_path, 1, keep={files[3]})
    assert {p.name for p in tmp_path.iterdir()} == {files[0].name, files[3].name}


def test_eviction_cleans_only_stale_temp_files(tmp_path):
    stale = _touch(tmp_path / f"prefix-abc.123{TMP_SUFFIX}", age_s=7200)
    young = _touch(tmp_path / f"prefix-def.456{TMP_SUFFIX}", age_s=5)
    other = _touch(tmp_path / "notes.txt", age_s=99999)
    evict_cache_files(tmp_path, 8)
    assert not stale.exists() and young.exists() and other.exists()


# ---------------------------------------------------------------- quantized KV (kv_bits)
def _fingerprint_of(tmp_path, **kw):
    from narrow_ai.models.kv_cache import PrefixKVCache

    kv = PrefixKVCache(object(), mode="persistent", cache_dir=tmp_path,
                       base_fingerprint={"model_id": "m"}, **kw)
    seen = []
    kv._try_load = lambda fp: seen.append(fp) or True  # record instead of touching MLX
    kv.prepare("system prompt", [1, 2, 3])
    return seen[0]


def test_kv_bits_get_their_own_disk_cache(tmp_path):
    full = _fingerprint_of(tmp_path)
    q8 = _fingerprint_of(tmp_path, kv_bits=8)
    q4 = _fingerprint_of(tmp_path, kv_bits=4)
    assert len({full["fingerprint"], q8["fingerprint"], q4["fingerprint"]}) == 3
    assert "kv_bits" not in full  # existing full-precision caches keep their names
    assert q8["kv_bits"] == "8"


def test_invalid_kv_bits_fail_closed():
    from narrow_ai.models.kv_cache import PrefixKVCache
    from narrow_ai.settings import ModelSettings

    with pytest.raises(ValueError):
        PrefixKVCache(object(), kv_bits=3)
    with pytest.raises(ValueError):
        ModelSettings(kv_bits=2)


def test_kv_bits_with_rotating_window_is_refused():
    pytest.importorskip("mlx_lm")
    from narrow_ai.models.mlx_backend import MlxBackend

    with pytest.raises(ValueError, match="kv_bits"):
        MlxBackend("m", kv_bits=8, max_kv_size=512, load=False)

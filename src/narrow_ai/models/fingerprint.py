"""Content fingerprints for the model, tokenizer and adapter behind a KV cache.

A persisted prefix KV cache is only reusable for *identical* parameters,
tokenizer and chat template. A repo id like ``mlx-community/Qwen3-1.7B-4bit``
is not enough: the local snapshot can change under the same id (a newer
download, an edited local directory, a retrained adapter).

For Hugging Face cache snapshots this is cheap: ``…/snapshots/<commit>/`` names
the revision, and every file is a symlink to ``…/blobs/<hash>``, where the blob
name is the file's content hash (SHA-256 for LFS files such as the weights, git
SHA-1 for small files). Reading the link target costs no I/O over the weights.

For plain local directories, files up to ``HASH_LIMIT`` bytes are SHA-256'd.
Larger files fall back to ``size:mtime_ns``, which is weaker but still catches
any normal rewrite. Adapters are always hashed in full, since LoRA files are small.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path

HASH_LIMIT = 64 * 1024 * 1024
FP_VERSION = "2"

WEIGHT_PATTERNS = ("*.safetensors", "*.safetensors.index.json", "*.npz", "*.gguf")
TOKENIZER_PATTERNS = (
    "tokenizer*.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "chat_template*",
)
CONFIG_PATTERNS = ("config.json", "generation_config.json")


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def file_digest(path: Path, *, hash_limit: int = HASH_LIMIT) -> str:
    """Content identity of one file (blob name for HF symlinks, else a hash)."""
    if path.is_symlink():
        target = os.readlink(path)
        blob = Path(target).name
        if "blobs" in Path(target).parts and blob:
            return f"blob:{blob}"
        path = path.resolve()
    st = path.stat()
    if st.st_size > hash_limit:
        return f"stat:{st.st_size}:{st.st_mtime_ns}"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _group_digest(root: Path, patterns: tuple[str, ...], *, hash_limit: int) -> str:
    entries = []
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_dir() and not p.is_symlink():
            continue
        if _matches(p.name, patterns):
            entries.append(f"{p.name}={file_digest(p, hash_limit=hash_limit)}")
    if not entries:
        return "none"
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:32]


def snapshot_revision(source: Path) -> str:
    """HF commit hash if ``source`` is ``…/snapshots/<commit>``, else ``local``."""
    if source.parent.name == "snapshots" and len(source.name) >= 7:
        return source.name
    return "local"


def snapshot_fingerprint(source: str | Path, *, hash_limit: int = HASH_LIMIT) -> dict[str, str]:
    """Revision + content digests of a resolved model directory.

    ``source`` is what the loader actually read (a local snapshot path). If it
    isn't a directory (the model will be downloaded), the fingerprint says so,
    and the cache is then keyed on the repo id alone.
    """
    root = Path(source).expanduser()
    if not root.is_dir():
        return {"revision": "unresolved", "weights_sha": "unresolved",
                "tokenizer_sha": "unresolved", "config_sha": "unresolved"}
    return {
        "revision": snapshot_revision(root),
        "weights_sha": _group_digest(root, WEIGHT_PATTERNS, hash_limit=hash_limit),
        "tokenizer_sha": _group_digest(root, TOKENIZER_PATTERNS, hash_limit=hash_limit),
        "config_sha": _group_digest(root, CONFIG_PATTERNS, hash_limit=hash_limit),
    }


def adapter_fingerprint(path: str | Path | None) -> str:
    """SHA-256 over a LoRA adapter's loaded files (or every file / the file itself)."""
    if not path:
        return "none"
    root = Path(path).expanduser()
    if not root.exists():
        return "missing"
    if root.is_dir():
        # mlx-lm reads exactly these two; training checkpoints beside them don't matter.
        loaded = [root / "adapters.safetensors", root / "adapter_config.json"]
        files = [f for f in loaded if f.is_file()] or sorted(
            p for p in root.rglob("*") if p.is_file()
        )
    else:
        files = [root]
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(root).as_posix() if root.is_dir() else f.name
        h.update(rel.encode() + b"\0" + file_digest(f, hash_limit=1 << 62).encode() + b"\n")
    return h.hexdigest()[:32]

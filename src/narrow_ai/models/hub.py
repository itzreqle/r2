"""Resolve model ids to local snapshot paths so R2 works fully offline once cached."""

from __future__ import annotations

from pathlib import Path


def local_or_repo(model_id: str) -> str:
    """Local path if ``model_id`` is a directory or already in the HF cache; else the id."""
    if Path(model_id).expanduser().exists():
        return str(Path(model_id).expanduser())
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet → let the loader download it
        return model_id

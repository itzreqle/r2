"""Select backend from settings without coupling the app to one checkpoint."""

from __future__ import annotations

from narrow_ai.models.base import ModelBackend
from narrow_ai.models.fake_backend import FakeModelBackend
from narrow_ai.settings import Settings


def create_backend(settings: Settings, *, load_mlx: bool = True) -> ModelBackend:
    backend = settings.model.backend.lower().strip()
    if backend == "fake":
        return FakeModelBackend()
    if backend == "mlx":
        from narrow_ai.models.mlx_backend import MlxBackend

        return MlxBackend(
            settings.model.model_id,
            adapter_path=settings.adapter_path(),
            max_tokens=settings.model.max_tokens,
            temperature=settings.model.temperature,
            max_kv_size=settings.model.max_kv_size or None,
            cache_mode=settings.model.cache_mode,
            cache_dir=settings.kv_cache_dir(),
            cache_max_files=settings.model.kv_cache_max_files,
            load=load_mlx,
        )
    raise ValueError(f"Unknown model backend: {backend}")

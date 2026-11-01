"""Local speech-to-text via MLX Whisper when available."""

from __future__ import annotations

from pathlib import Path


def transcribe_wav(path: Path, *, model: str = "mlx-community/whisper-tiny") -> str:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper not installed. uv sync --group speech  (Apple Silicon)"
        ) from exc
    result = mlx_whisper.transcribe(str(path), path_or_hf_repo=model)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()

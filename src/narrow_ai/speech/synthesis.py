"""Text-to-speech abstraction with macOS `say` fallback."""

from __future__ import annotations

import shutil
import subprocess


def synthesize_speech(text: str, *, backend: str = "macos_say", voice: str | None = None) -> None:
    text = text.strip()
    if not text:
        return
    if backend in {"macos_say", "say"}:
        _macos_say(text, voice=voice)
        return
    if backend in {"kokoro", "kokoro_mlx"}:
        try:
            _kokoro_mlx(text, voice=voice)
            return
        except Exception:
            _macos_say(text, voice=voice)
            return
    raise ValueError(f"Unknown TTS backend: {backend}")


def _macos_say(text: str, *, voice: str | None) -> None:
    if not shutil.which("say"):
        raise RuntimeError("macOS `say` not found")
    cmd = ["say"]
    if voice:
        cmd.extend(["-v", voice])
    cmd.append(text)
    subprocess.run(cmd, check=False, timeout=120)


def _kokoro_mlx(text: str, *, voice: str | None) -> None:
    # Optional dependency — verify package presence at call time.
    try:
        import kokoro_mlx  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Kokoro-MLX not installed; using fallback") from exc
    raise RuntimeError("Kokoro-MLX integration pending verified API; use macos_say")

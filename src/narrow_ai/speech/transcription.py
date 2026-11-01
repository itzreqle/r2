"""Local speech-to-text via MLX Whisper (model stays cached in-process)."""

from __future__ import annotations

from pathlib import Path


# Biases Whisper toward R2's vocabulary (wake name, git terms, app names).
VOCAB_PROMPT = (
    "Hey R2. Git status, git diff, branch, commit, README, pyproject.toml, Python, Swift, "
    "TypeScript, markdown, Cursor, Xcode, Visual Studio Code, Terminal."
)


def transcribe_wav(
    path: Path,
    *,
    model: str = "mlx-community/whisper-small-mlx",
    language: str | None = "en",
    initial_prompt: str | None = VOCAB_PROMPT,
) -> str:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError(
            "mlx-whisper not installed. uv sync --group speech  (Apple Silicon)"
        ) from exc
    from narrow_ai.models.hub import local_or_repo

    kwargs = {"path_or_hf_repo": local_or_repo(model)}
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    result = mlx_whisper.transcribe(str(path), **kwargs)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


def synthesize_to_wav(text: str, out: Path, *, voice: str | None = None) -> Path:
    """Use macOS `say` to create a 16 kHz WAV (handy for testing STT without a mic)."""
    import subprocess

    aiff = out.with_suffix(".aiff")
    cmd = ["say", "-o", str(aiff)]
    if voice:
        cmd += ["-v", voice]
    subprocess.run([*cmd, text], check=True, timeout=60)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(out)],
        check=True,
        timeout=60,
    )
    aiff.unlink(missing_ok=True)
    return out

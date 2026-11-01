"""Push-to-talk microphone capture."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np


def record_push_to_talk(
    *,
    sample_rate: int = 16000,
    max_seconds: float = 12.0,
) -> Path:
    """Record until Enter is pressed again (blocking CLI helper)."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install speech extras: uv sync --group speech") from exc

    print("Recording... press Enter to stop.")
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time, status):  # noqa: ARG001
        if status:
            print(status)
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    if not chunks:
        raise RuntimeError("No audio captured")
    audio = np.concatenate(chunks, axis=0)
    max_samples = int(sample_rate * max_seconds)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    out = Path(tempfile.mkstemp(suffix=".wav", prefix="narrow-ai-")[1])
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out

"""Text-to-speech (macOS `say`) and Siri-style chimes."""

from __future__ import annotations

import shutil
import subprocess

_CHIMES = {
    "listen": "/System/Library/Sounds/Tink.aiff",
    "done": "/System/Library/Sounds/Pop.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
}


def chime(kind: str) -> None:
    """Play a short non-blocking system sound."""
    path = _CHIMES.get(kind)
    if path and shutil.which("afplay"):
        subprocess.Popen(  # noqa: S603 - fixed system binary + fixed path
            ["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def synthesize_speech(
    text: str, *, backend: str = "macos_say", voice: str | None = None, block: bool = True
) -> subprocess.Popen | None:
    text = text.strip()
    if not text:
        return None
    if backend not in {"macos_say", "say"}:
        raise ValueError(f"Unknown TTS backend: {backend}")
    if not shutil.which("say"):
        raise RuntimeError("macOS `say` not found")
    cmd = ["say"]
    if voice:
        cmd.extend(["-v", voice])
    cmd.append(text)
    if block:
        subprocess.run(cmd, check=False, timeout=120)
        return None
    return subprocess.Popen(cmd)

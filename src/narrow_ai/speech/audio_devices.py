"""Audio device discovery helpers."""

from __future__ import annotations


def list_input_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install speech extras: uv sync --group speech") from exc
    devices = sd.query_devices()
    out = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d.get("name"), "channels": d["max_input_channels"]})
    return out

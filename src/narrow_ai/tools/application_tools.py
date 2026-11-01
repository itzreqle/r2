"""Open approved macOS applications via `open -a`."""

from __future__ import annotations

import subprocess


class AppToolError(RuntimeError):
    pass


def open_approved_app(app_name: str, approved: list[str]) -> str:
    if app_name not in approved:
        raise AppToolError(f"App not on allowlist: {app_name}")
    proc = subprocess.run(
        ["open", "-a", app_name],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise AppToolError((proc.stderr or "open failed").strip())
    return f"Opened {app_name}"

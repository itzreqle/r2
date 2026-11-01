"""Per-session bearer token for the local R2 HTTP API.

``r2 serve`` writes a fresh random token to a 0600 file before it starts
listening and deletes it on shutdown. Clients on the same machine (the SwiftUI
app) read the file and send ``Authorization: Bearer <token>``.

Trust boundary, stated honestly: any process running as the same macOS user can
read a 0600 file, so it stays inside the boundary (it could equally read the
preference DB or edit R2's config). What the token does stop is web pages in a
browser (CSRF / DNS rebinding), other local users, and accidental clients.
"""

from __future__ import annotations

import contextlib
import hmac
import os
import secrets
from pathlib import Path


class ApiToken:
    def __init__(self, value: str, path: Path | None = None) -> None:
        if len(value) < 32:
            raise ValueError("API token too short")
        self.value = value
        self.path = path

    @classmethod
    def create(cls, path: Path) -> ApiToken:
        """Generate a token and write it atomically to ``path`` with mode 0600."""
        value = secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)  # ignore a permissive umask
            os.write(fd, value.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        return cls(value, path)

    def verify(self, authorization: str | None) -> bool:
        """Constant-time check of an ``Authorization: Bearer <token>`` header."""
        if not authorization:
            return False
        scheme, _, presented = authorization.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            return False
        return hmac.compare_digest(presented.strip().encode(), self.value.encode())

    def remove(self) -> None:
        """Delete the token file if it still holds *this* session's token."""
        if self.path is None:
            return
        with contextlib.suppress(OSError):
            if self.path.read_text().strip() == self.value:
                self.path.unlink()

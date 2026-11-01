"""Local SQLite key-value preference store."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class PreferenceStore:
    def __init__(self, db_path: Path, *, max_preferences: int = 100) -> None:
        self.db_path = db_path
        self.max_preferences = max_preferences
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM preferences").fetchone()["c"]
            exists = conn.execute(
                "SELECT 1 FROM preferences WHERE key = ?", (key,)
            ).fetchone()
            if exists is None and count >= self.max_preferences:
                raise RuntimeError("Preference store full; clear some entries first")
            conn.execute(
                """
                INSERT INTO preferences(key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )
            conn.commit()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()
            return None if row is None else str(row["value"])

    def delete(self, key: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
            conn.commit()
            return cur.rowcount > 0

    def list_keys(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key FROM preferences ORDER BY key"
            ).fetchall()
            return [str(r["key"]) for r in rows]

    def clear(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM preferences")
            conn.commit()
            return cur.rowcount

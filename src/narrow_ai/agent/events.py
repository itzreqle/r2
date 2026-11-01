"""Typed run events, an in-memory bus, and metadata-only SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_RESULT = "step_result"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    RUN_FINISHED = "run_finished"


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    type: EventType
    at: float  # wall clock (time.time) for display/ordering
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = str(self.type)
        return d


class EventBus:
    """Keeps the last ``keep`` events per run and fans out to subscribers."""

    def __init__(self, keep: int = 200) -> None:
        self._history: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=keep))
        self._seq: dict[str, int] = defaultdict(int)
        self._subs: list[Callable[[Event], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subs.append(fn)

    def publish(self, run_id: str, type: EventType, **data: Any) -> Event:
        with self._lock:
            self._seq[run_id] += 1
            ev = Event(run_id, self._seq[run_id], type, time.time(), data)
            self._history[run_id].append(ev)
        for fn in list(self._subs):
            fn(ev)
        return ev

    def history(self, run_id: str) -> list[Event]:
        return list(self._history.get(run_id, ()))

    def forget(self, run_id: str) -> None:
        with self._lock:
            self._history.pop(run_id, None)
            self._seq.pop(run_id, None)


class RunStore:
    """SQLite record of runs and events.

    Stores *metadata only*: the goal text, statuses, step intents and parameters,
    ok/failed, and the short spoken summary. Tool output (file contents, diffs,
    answers' full text) is never written here; see SECURITY.md "Data retention".
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, steps TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL,
                    at REAL NOT NULL, data TEXT NOT NULL, PRIMARY KEY (run_id, seq)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save_run(self, run: dict[str, Any]) -> None:
        steps = json.dumps(run["steps"], ensure_ascii=False)
        now = time.time()
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO runs (id, goal, status, created, updated, steps, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "status=excluded.status, updated=excluded.updated, steps=excluded.steps, "
                "error=excluded.error",
                (run["id"], run["goal"], run["status"], now, now, steps, run.get("error", "")),
            )

    def save_event(self, ev: Event) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO run_events VALUES (?, ?, ?, ?, ?)",
                (ev.run_id, ev.seq, str(ev.type), ev.at, json.dumps(ev.data, ensure_ascii=False)),
            )

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, goal, status, steps, error FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "goal": row[1], "status": row[2], "steps": json.loads(row[3]),
                "error": row[4]}

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT seq, type, at, data FROM run_events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [{"run_id": run_id, "seq": s, "type": t, "at": a, "data": json.loads(d)}
                for s, t, a, d in rows]

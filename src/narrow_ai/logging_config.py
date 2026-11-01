"""Structured logging setup."""

from __future__ import annotations

import logging
import re
import sys

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+"
)


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SECRET_RE.sub(r"\1=[REDACTED]", record.msg)
        return True


def setup_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        fmt = '{"level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}'
    else:
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    handler.addFilter(_RedactFilter())
    root.addHandler(handler)

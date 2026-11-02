"""Run an allow-listed test command and summarise its output.

Safety model:

* The model only chooses a *command id*. The argv comes from ``tools.test_commands``
  in the config and is executed directly, never through a shell.
* Shells and interpreters-with-inline-code as ``argv[0]`` are refused, so the config
  cannot smuggle in ``sh -c …`` by accident.
* The run is bounded: wall-clock timeout (the whole process group is killed), no stdin,
  and only the head and tail of the output are kept in memory.
* Running tests executes the project's own code, so the gate always asks for
  confirmation first (``PermissionGate``).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

HEAD_CHARS = 4_000
TAIL_CHARS = 60_000
EXCERPT_CHARS = 3_000

# argv[0] basenames that would turn a fixed argv back into "run any command".
FORBIDDEN_LAUNCHERS = {
    "sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh",
    "sudo", "doas", "su", "env", "xargs", "osascript", "open", "eval", "exec",
}


class TestToolError(RuntimeError):
    __test__ = False  # not a pytest test class


@dataclass
class TestRun:
    __test__ = False  # not a pytest test class

    command_id: str
    argv: list[str]
    exit_code: int | None
    duration_sec: float
    output: str
    timed_out: bool = False
    truncated: bool = False
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def resolve_command(test_commands: dict[str, list[str]], command_id: str) -> tuple[str, list[str]]:
    """Map a (possibly empty) id to its configured argv. Raises TestToolError."""
    if not test_commands:
        raise TestToolError("no test commands are configured")
    cid = command_id.strip() or next(iter(test_commands))
    if cid not in test_commands:
        raise TestToolError(
            f"unknown test command '{cid}'. Configured: {', '.join(test_commands)}"
        )
    argv = test_commands[cid]
    validate_argv(argv)
    return cid, list(argv)


def validate_argv(argv: list[str]) -> None:
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise TestToolError("test command must be a non-empty list of strings")
    exe = Path(argv[0]).name
    if exe in FORBIDDEN_LAUNCHERS:
        raise TestToolError(f"'{exe}' is not allowed as a test command launcher")
    if exe.startswith("python") and any(a in {"-c", "-"} for a in argv[1:2]):
        raise TestToolError("inline interpreter code is not allowed in a test command")
    if any("\x00" in a for a in argv):
        raise TestToolError("test command contains a NUL byte")
    if shutil.which(argv[0]) is None:
        raise TestToolError(f"'{argv[0]}' was not found on PATH")


class _Collector:
    """Keep the first HEAD_CHARS and the last TAIL_CHARS characters of a stream."""

    def __init__(self) -> None:
        self.head: list[str] = []
        self.head_len = 0
        self.tail: deque[str] = deque()
        self.tail_len = 0
        self.dropped = False

    def add(self, chunk: str) -> None:
        if self.head_len < HEAD_CHARS:
            take = chunk[: HEAD_CHARS - self.head_len]
            self.head.append(take)
            self.head_len += len(take)
            chunk = chunk[len(take):]
        if not chunk:
            return
        self.tail.append(chunk)
        self.tail_len += len(chunk)
        while self.tail_len > TAIL_CHARS and self.tail:
            first = self.tail[0]
            over = self.tail_len - TAIL_CHARS
            if len(first) <= over:
                self.tail.popleft()
                self.tail_len -= len(first)
            else:
                self.tail[0] = first[over:]
                self.tail_len -= over
            self.dropped = True

    def text(self) -> str:
        head, tail = "".join(self.head), "".join(self.tail)
        return head + ("\n…[output truncated]…\n" if self.dropped else "") + tail


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue


def run_test_command(
    workspace: Path, command_id: str, argv: list[str], *, timeout: int = 300
) -> TestRun:
    validate_argv(argv)
    env = dict(os.environ)
    env.update({"CI": "1", "NO_COLOR": "1", "TERM": "dumb", "PYTHONUNBUFFERED": "1"})
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv from config, no shell
            argv,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # own process group, so a timeout kills children too
        )
    except OSError as exc:
        raise TestToolError(f"could not start test command: {exc}") from exc
    col = _Collector()

    def pump() -> None:
        assert proc.stdout is not None
        for raw in iter(lambda: proc.stdout.read(8192), b""):
            col.add(raw.decode("utf-8", "replace"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
    reader.join(timeout=5)
    output = col.text()
    return TestRun(
        command_id=command_id,
        argv=argv,
        exit_code=proc.returncode,
        duration_sec=time.monotonic() - t0,
        output=output,
        timed_out=timed_out,
        truncated=col.dropped,
        counts=parse_counts(output),
    )


# --------------------------------------------------------------------- understanding output
_PYTEST = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")
_SWIFT = re.compile(r"Executed (\d+) tests?, with (\d+) failures?")
_JEST = re.compile(
    r"Tests:\s+(?:(\d+) failed, )?(?:(\d+) skipped, )?(?:(\d+) passed, )?(\d+) total"
)
_GO_FAIL = re.compile(r"^--- FAIL: (\S+)", re.MULTILINE)
_FAILED_ID = re.compile(r"^(?:FAILED|ERROR) (\S+)(?: - (.*))?$", re.MULTILINE)
_SWIFT_FAIL = re.compile(r"error: -\[(\S+ \S+)\] : (.*)$|Test Case '(.+)' failed", re.MULTILINE)


def parse_counts(output: str) -> dict[str, int]:
    """Best-effort pass/fail counts for pytest, XCTest/swift test, Jest and go test."""
    counts: dict[str, int] = {}
    tail = output[-6_000:]
    lines = [ln for ln in tail.splitlines() if _PYTEST.search(ln)]
    if lines:
        for n, kind in _PYTEST.findall(lines[-1]):
            counts["errors" if kind.startswith("error") else kind] = int(n)
        return counts
    if m := list(_SWIFT.finditer(tail)):
        total, failed = int(m[-1].group(1)), int(m[-1].group(2))
        return {"passed": total - failed, "failed": failed}
    if m := list(_JEST.finditer(tail)):
        failed, skipped, passed, _total = (int(g) if g else 0 for g in m[-1].groups())
        return {k: v for k, v in
                {"passed": passed, "failed": failed, "skipped": skipped}.items() if v}
    if fails := _GO_FAIL.findall(output):
        counts["failed"] = len(fails)
    return counts


def failing_tests(output: str, limit: int = 5) -> list[str]:
    names: list[str] = []
    for m in _FAILED_ID.finditer(output):
        names.append(m.group(1) + (f" ({m.group(2).strip()[:120]})" if m.group(2) else ""))
    for m in _SWIFT_FAIL.finditer(output):
        names.append((m.group(1) or m.group(3) or "").strip())
    names += _GO_FAIL.findall(output)
    seen: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.append(n)
    return seen[:limit]


def failure_excerpt(output: str, limit: int = EXCERPT_CHARS) -> str:
    """The part of the output most worth explaining: the failure section if there is
    one (pytest/XCTest/Jest markers), otherwise the tail."""
    markers = ("= FAILURES =", "= ERRORS =", "short test summary", "● ", "--- FAIL",
               "error: -[", "Traceback (most recent call last)")
    starts = [output.find(m) for m in markers if m in output]
    if starts:
        first = min(starts)
        text = output[output.rfind("\n", 0, first) + 1:]  # from the start of that line
    else:
        text = output
    if len(text) > limit:
        text = text[: limit // 3] + "\n…\n" + text[-(limit - limit // 3):]
    return text.strip()


def summarize(run: TestRun) -> str:
    """One spoken sentence."""
    c = run.counts
    secs = f"{run.duration_sec:.1f} seconds"
    if run.timed_out:
        return f"The {run.command_id} tests timed out after {secs} and were stopped."
    failed = c.get("failed", 0) + c.get("errors", 0)
    if run.passed:
        if c.get("passed"):
            return f"All {c['passed']} {run.command_id} tests passed in {secs}."
        return f"The {run.command_id} tests passed in {secs}."
    if failed:
        total = failed + c.get("passed", 0)
        return f"{failed} of {total} {run.command_id} tests failed."
    return f"The {run.command_id} tests failed with exit code {run.exit_code}."


def heuristic_explanation(run_output: str) -> str:
    """Deterministic fallback (no model): name the failing tests and the first error."""
    names = failing_tests(run_output, limit=3)
    err = ""
    for line in run_output.splitlines():
        s = line.strip()
        if s.startswith(("E ", "AssertionError", "Error:", "error:")) or "Error:" in s[:40]:
            err = s.removeprefix("E").strip()[:160]
            break
    if names and err:
        return f"Failing: {', '.join(names)}. First error: {err}"
    if names:
        return f"Failing: {', '.join(names)}."
    if err:
        return f"First error: {err}"
    return "I couldn't find a clear failure in the output."


# Backwards-compatible helper used by older callers/tests.
def run_allowlisted_test(
    workspace: Path, command: list[str], *, timeout: int = 120, max_chars: int = 8000
) -> str:
    run = run_test_command(workspace, "custom", command, timeout=timeout)
    if run.timed_out:
        raise TestToolError("test command timed out")
    return f"exit={run.exit_code}\n{run.output}"[:max_chars]

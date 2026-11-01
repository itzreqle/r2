"""`r2` — Siri-style terminal app.

r2                 chat (type, or press Enter on an empty line to talk)
r2 ask "..."       one request, then exit
r2 serve           local HTTP API for the SwiftUI menu-bar client
r2 status          model + KV cache info
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from narrow_ai.logging_config import setup_logging

app = typer.Typer(
    name="r2",
    help="R2 — local, Siri-style developer assistant (MLX + KV cache).",
    invoke_without_command=True,
    add_completion=False,
)
console = Console()

ORB = {
    "idle": ("◉", "cyan"),
    "listening": ("◉", "red"),
    "thinking": ("◌", "magenta"),
    "awaiting_confirmation": ("◉", "yellow"),
    "error": ("◉", "red"),
}

HELP = """[bold]Talk to R2[/bold]
  type a request          e.g. "what's my git status", "find swift files", "open Cursor"
  [bold]Enter[/bold] on empty line    push-to-talk (Enter again to stop)
  yes / no                answer a confirmation
  /status  /reset  /mute  /unmute  /help  /quit"""


def _build(backend: str | None, workspace: Path | None, cache_mode: str | None):
    from narrow_ai.r2.core import R2
    from narrow_ai.settings import load_settings

    settings = load_settings()
    setup_logging(settings.logging.level, json_logs=settings.logging.json_logs)
    with console.status("[magenta]R2 is waking up… (loading model + KV cache)[/magenta]"):
        return R2(settings, backend=backend, workspace=workspace, cache_mode=cache_mode)


def _banner(r2) -> Panel:
    st = r2.status()
    kv = st["kv_cache"]
    if kv["mode"] == "none":
        kv_line = "KV cache: off"
    else:
        how = (
            f"loaded from disk in {kv['load_ms']:.0f} ms"
            if kv["source"] == "disk"
            else (f"built in {kv['build_ms']:.0f} ms")
        )
        kv_line = f"KV cache: {kv['mode']} · {kv['prefix_tokens']} prompt tokens {how}"
    body = Text.assemble(
        ("◉ ", "bold cyan"),
        ("R2 is ready\n", "bold"),
        (f"model {st['model']} · loaded in {st['load_ms'] / 1000:.1f}s\n", "dim"),
        (kv_line + "\n", "dim"),
        (f"workspace {st['workspace']}\n", "dim"),
        ("Type a request, or press Enter on an empty line to talk. /help", "italic"),
    )
    return Panel(body, title="[bold cyan]R2[/bold cyan]", border_style="cyan", expand=False)


def _metrics_line(resp) -> str:
    m = resp.metrics or {}
    if not m:
        return ""
    parts = [f"intent={resp.decision.intent.value}" if resp.decision else ""]
    if m.get("backend") == "mlx":
        parts.append(f"ttft {m.get('ttft_ms', 0):.0f}ms")
        parts.append(f"total {m.get('think_ms', 0):.0f}ms")
        parts.append(f"{m.get('cached_tokens', 0)} cached + {m.get('prompt_tokens', 0)} new tokens")
    if m.get("source") not in (None, "model"):
        parts.append(f"via {m.get('source')}")
    return " · ".join(p for p in parts if p)


def _show(resp, *, heard: str | None = None) -> None:
    state = (
        "awaiting_confirmation"
        if resp.needs_confirmation
        else ("idle" if (resp.tool_result is None or resp.tool_result.ok) else "error")
    )
    icon, color = ORB.get(state, ORB["idle"])
    items = []
    if resp.speech and resp.speech != resp.text and not resp.needs_confirmation:
        items.append(Text(resp.speech, style="bold"))
        items.append(Text(resp.text, style="white"))
    elif resp.needs_confirmation:
        items.append(Text(resp.speech or resp.text, style="bold"))
    else:
        items.append(Text(resp.text, style="bold" if len(resp.text) < 200 else "white"))
    line = _metrics_line(resp)
    if line:
        items.append(Text(line, style="dim"))
    console.print(
        Panel(
            Group(*items),
            title=f"[{color}]{icon} R2[/{color}]",
            title_align="left",
            border_style=color,
        )
    )


class _Speaker:
    """Non-blocking TTS; a new utterance interrupts the previous one (like Siri)."""

    def __init__(self, r2, enabled: bool) -> None:
        self.r2 = r2
        self.enabled = enabled
        self.proc: subprocess.Popen | None = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        from narrow_ai.speech.synthesis import synthesize_speech

        self.stop()
        try:
            self.proc = synthesize_speech(
                text, voice=self.r2.settings.speech.voice or None, block=False
            )
        except Exception:  # noqa: BLE001
            self.proc = None


def _listen(r2) -> str:
    try:
        r2.listen_start()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Microphone unavailable:[/red] {exc}")
        return ""
    console.print("[bold red]◉ Listening…[/bold red] [dim]press Enter to stop[/dim]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    try:
        with console.status("[magenta]Transcribing…[/magenta]"):
            wav = r2._recorder.stop()
            r2.chime("done")
            text = r2.transcribe(wav)
            wav.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Voice error:[/red] {exc}")
        return ""
    return text


@app.callback()
def main(
    ctx: typer.Context,
    backend: str | None = typer.Option(None, "--backend", "-b", help="mlx | fake"),
    workspace: Path | None = typer.Option(None, "--workspace", "-w", help="Project folder"),
    cache_mode: str | None = typer.Option(None, "--cache", help="none | prefix | persistent"),
    mute: bool = typer.Option(False, "--mute", help="Do not speak replies"),
) -> None:
    """Start the interactive R2 chat when no sub-command is given."""
    ctx.obj = {"backend": backend, "workspace": workspace, "cache_mode": cache_mode, "mute": mute}
    if ctx.invoked_subcommand is None:
        chat(backend, workspace, cache_mode, mute)


def chat(backend, workspace, cache_mode, mute) -> None:
    r2 = _build(backend, workspace, cache_mode)
    speaker = _Speaker(r2, enabled=(not mute) and r2.settings.speech.speak_replies)
    console.print(_banner(r2))
    speaker.say("R2 ready.")
    while True:
        try:
            line = console.input("[bold green]you ▸ [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        speaker.stop()
        heard = None
        if not line:
            line = _listen(r2)
            if not line:
                console.print("[yellow]I didn't catch that.[/yellow]")
                continue
            heard = line
            console.print(f"[bold green]you 🎙 [/bold green]{line}")
        cmd = line.lower()
        if cmd in {"/quit", "/exit", "quit", "exit", "/q"}:
            break
        if cmd == "/help":
            console.print(Panel(HELP, border_style="cyan", expand=False))
            continue
        if cmd == "/status":
            console.print_json(json.dumps(r2.status()))
            continue
        if cmd == "/reset":
            r2.reset()
            console.print("[dim]conversation reset[/dim]")
            continue
        if cmd in {"/mute", "/unmute"}:
            speaker.enabled = cmd == "/unmute"
            console.print(f"[dim]voice {'on' if speaker.enabled else 'off'}[/dim]")
            continue
        with console.status("[magenta]◌ R2 is thinking…[/magenta]"):
            resp = r2.ask(line)
        _show(resp, heard=heard)
        speaker.say(resp.spoken_text())
    speaker.stop()


@app.command()
def ask(
    ctx: typer.Context,
    text: str = typer.Argument(..., help='Request, e.g. "what\'s my git status"'),
    speak: bool = typer.Option(False, "--speak", help="Speak the reply"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON"),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm privileged actions"),
) -> None:
    """Run a single request and exit."""
    o = ctx.obj or {}
    r2 = _build(o.get("backend"), o.get("workspace"), o.get("cache_mode"))
    resp = r2.ask(text)
    if yes and resp.needs_confirmation:
        resp = r2.confirm()
    if as_json:
        console.print_json(json.dumps(resp.to_dict()))
    else:
        _show(resp)
    if speak:
        r2.speak(resp.spoken_text())
    raise typer.Exit(0 if resp.decision else 1)


@app.command()
def say(
    ctx: typer.Context,
    phrase: str = typer.Argument(..., help="Text to synthesize, transcribe and send to R2"),
) -> None:
    """Voice-pipeline self test without a microphone: TTS → WAV → Whisper → R2."""
    import tempfile

    from narrow_ai.speech.transcription import synthesize_to_wav

    o = ctx.obj or {}
    r2 = _build(o.get("backend"), o.get("workspace"), o.get("cache_mode"))
    wav = synthesize_to_wav(phrase, Path(tempfile.mkdtemp()) / "say.wav")
    with console.status("[magenta]Transcribing…[/magenta]"):
        heard, resp = r2.ask_audio(wav)
    console.print(f"[bold green]heard 🎙 [/bold green]{heard}")
    if resp:
        _show(resp)


@app.command()
def status(ctx: typer.Context) -> None:
    """Show model and KV cache status."""
    o = ctx.obj or {}
    r2 = _build(o.get("backend"), o.get("workspace"), o.get("cache_mode"))
    console.print_json(json.dumps(r2.status()))


@app.command()
def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", help="Bind address (keep it local)"),
    port: int = typer.Option(8765, help="Port"),
) -> None:
    """Run the local R2 HTTP API (used by the SwiftUI menu-bar app)."""
    from narrow_ai.r2.server import serve_forever

    o = ctx.obj or {}
    r2 = _build(o.get("backend"), o.get("workspace"), o.get("cache_mode"))
    with console.status("[magenta]Warming up the neural voice…[/magenta]"):
        voice_ok = r2.warmup_tts()
    console.print(
        f"[dim]neural voice: {'Kokoro ready' if voice_ok else 'unavailable (system voice)'}[/dim]"
    )
    console.print(_banner(r2))
    console.print(f"[cyan]R2 API listening on http://{host}:{port}[/cyan]  (Ctrl+C to stop)")
    serve_forever(r2, host=host, port=port, speak=not o.get("mute"))


def run() -> None:  # console-script entry point
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    app()


if __name__ == "__main__":
    run()

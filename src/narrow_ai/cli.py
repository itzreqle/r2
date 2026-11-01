"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from narrow_ai.assistant.orchestrator import AssistantOrchestrator
from narrow_ai.logging_config import setup_logging
from narrow_ai.models.model_registry import create_backend
from narrow_ai.settings import load_settings

app = typer.Typer(name="narrow-ai", help="Local narrow macOS developer assistant", no_args_is_help=True)
console = Console()


def _build(workspace: Optional[Path], backend_name: Optional[str], load_mlx: bool) -> AssistantOrchestrator:
    settings = load_settings()
    if workspace is not None:
        settings.workspace.path = str(workspace.resolve())
    if backend_name:
        settings.model.backend = backend_name
    setup_logging(settings.logging.level, json_logs=settings.logging.json_logs)
    backend = create_backend(settings, load_mlx=load_mlx)
    return AssistantOrchestrator(settings, backend)


@app.command("text")
def text_mode(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Project directory"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="fake|mlx"),
    once: Optional[str] = typer.Option(None, "--once", "-c", help="Single command then exit"),
) -> None:
    """Interactive text assistant (no microphone)."""
    orch = _build(workspace, backend, load_mlx=(backend == "mlx"))
    console.print(
        Panel.fit(
            f"[bold]{orch.settings.assistant.name}[/bold] text mode\n"
            f"workspace: {orch.settings.workspace_path()}\n"
            f"backend: {orch.backend.name}\n"
            "Commands: quit | cancel | yes | no",
            title="narrow-ai",
        )
    )
    if once:
        resp = orch.handle_text(once)
        console.print(f"[cyan]intent[/cyan]: {resp.decision.intent if resp.decision else '-'}")
        console.print(resp.text)
        ok = bool(resp.tool_result and resp.tool_result.ok) or bool(resp.decision)
        raise typer.Exit(0 if ok else 1)

    while True:
        try:
            line = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            break
        if line.lower() == "cancel":
            orch.cancel()
            console.print("[yellow]cancelled[/yellow]")
            continue
        resp = orch.handle_text(line)
        if resp.decision:
            console.print(f"[dim]intent={resp.decision.intent} params={resp.decision.parameters}[/dim]")
        console.print(f"[bold]assistant>[/bold] {resp.text}")


@app.command("voice")
def voice_mode(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w"),
    backend: Optional[str] = typer.Option("fake", "--backend", "-b"),
) -> None:
    """Push-to-talk voice loop (requires speech extras)."""
    from narrow_ai.speech.input import record_push_to_talk
    from narrow_ai.speech.synthesis import synthesize_speech
    from narrow_ai.speech.transcription import transcribe_wav

    orch = _build(workspace, backend, load_mlx=(backend == "mlx"))
    console.print("[bold]Push-to-talk[/bold]: press Enter to record, speak, Enter to stop. `quit` to exit.")
    while True:
        cmd = console.input("[bold]ready>[/bold] ").strip().lower()
        if cmd in {"quit", "exit", "q"}:
            break
        try:
            wav = record_push_to_talk(sample_rate=orch.settings.speech.sample_rate)
            transcript = transcribe_wav(wav, model=orch.settings.speech.whisper_model)
            console.print(f"[cyan]heard:[/cyan] {transcript}")
            if not transcript.strip():
                console.print("[yellow]empty transcript[/yellow]")
                continue
            resp = orch.handle_text(transcript)
            console.print(f"[bold]assistant>[/bold] {resp.text}")
            synthesize_speech(resp.spoken_text(), backend=orch.settings.speech.tts_backend)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]voice error:[/red] {exc}")


@app.command("version")
def version() -> None:
    from narrow_ai import __version__

    console.print(__version__)


if __name__ == "__main__":
    app()

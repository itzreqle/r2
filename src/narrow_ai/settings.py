"""Typed settings loaded from config/default.toml and environment."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from narrow_ai.paths import ARTIFACTS_DIR, DEFAULT_CONFIG, REPO_ROOT


class AssistantSettings(BaseModel):
    name: str = "R2"
    domain: str = "macos_developer"
    system_prompt: str = ""


class WorkspaceSettings(BaseModel):
    path: str = ""


class ModelSettings(BaseModel):
    backend: str = "fake"
    model_id: str = "mlx-community/Qwen3-1.7B-4bit"
    adapter_path: str = ""
    max_tokens: int = 128
    temperature: float = 0.0
    max_history_turns: int = 3
    exclusive_heavy_jobs: bool = True
    max_kv_size: int = 0  # 0 = unbounded; >0 = rotating window (only when cache_mode=none)
    cache_mode: str = "prefix"  # none | prefix | persistent
    kv_cache_dir: str = ""  # empty = artifacts/kv
    kv_cache_max_files: int = 8  # persistent caches kept on disk (LRU by last use)
    constrained: bool = True  # constrain intent name during decoding
    few_shot: bool = True


class ToolsSettings(BaseModel):
    allow_git_status: bool = True
    allow_git_diff: bool = True
    allow_search_files: bool = True
    allow_read_file: bool = False
    allow_run_tests: bool = False
    allow_open_app: bool = False
    command_timeout_sec: int = 30
    max_output_chars: int = 8000
    approved_apps: list[str] = Field(default_factory=list)
    approved_test_commands: list[str] = Field(default_factory=list)


class MemorySettings(BaseModel):
    enabled: bool = True
    db_path: str = ""
    max_preferences: int = 100


class SpeechSettings(BaseModel):
    enabled: bool = False
    stt_backend: str = "mlx_whisper"
    tts_backend: str = "kokoro"  # kokoro (neural, natural) | macos_say
    whisper_model: str = "mlx-community/whisper-small-mlx"
    sample_rate: int = 16000
    voice: str = ""  # macOS `say` voice; empty = system default
    neural_voice: str = "af_heart"  # Kokoro voice
    speak_replies: bool = True
    chimes: bool = True


class LoggingSettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    level: str = "WARNING"
    json_logs: bool = Field(default=False, alias="json")


class SecuritySettings(BaseModel):
    require_confirmation_for_writes: bool = True
    block_path_escape: bool = True
    redact_secrets_in_logs: bool = True
    # Local HTTP API (r2 serve): bearer token file, rotated on every start (0600).
    api_token_path: str = ""  # empty = ~/.local/share/narrow-ai/api-token
    # Pending privileged actions expire after this many seconds.
    confirmation_ttl_sec: int = 120


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NARROW_AI_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    assistant: AssistantSettings = Field(default_factory=AssistantSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    tools: ToolsSettings = Field(default_factory=ToolsSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    speech: SpeechSettings = Field(default_factory=SpeechSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    def workspace_path(self) -> Path:
        raw = self.workspace.path.strip() or os.getcwd()
        return Path(raw).expanduser().resolve()

    def db_path(self) -> Path:
        if self.memory.db_path.strip():
            return Path(self.memory.db_path).expanduser().resolve()
        return Path.home() / ".local" / "share" / "narrow-ai" / "preferences.db"

    def api_token_path(self) -> Path:
        raw = self.security.api_token_path.strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return Path.home() / ".local" / "share" / "narrow-ai" / "api-token"

    def kv_cache_dir(self) -> Path:
        raw = self.model.kv_cache_dir.strip()
        return Path(raw).expanduser().resolve() if raw else ARTIFACTS_DIR / "kv"

    def adapter_path(self) -> Path | None:
        raw = self.model.adapter_path.strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    # Env overrides for common flat keys
    env_overlay: dict[str, Any] = {"model": {}, "workspace": {}, "memory": {}}
    if os.getenv("NARROW_AI_BACKEND"):
        env_overlay["model"]["backend"] = os.environ["NARROW_AI_BACKEND"]
    if os.getenv("NARROW_AI_MODEL_ID"):
        env_overlay["model"]["model_id"] = os.environ["NARROW_AI_MODEL_ID"]
    if os.getenv("NARROW_AI_ADAPTER_PATH"):
        env_overlay["model"]["adapter_path"] = os.environ["NARROW_AI_ADAPTER_PATH"]
    if os.getenv("NARROW_AI_WORKSPACE"):
        env_overlay["workspace"]["path"] = os.environ["NARROW_AI_WORKSPACE"]
    if os.getenv("NARROW_AI_DB_PATH"):
        env_overlay["memory"]["db_path"] = os.environ["NARROW_AI_DB_PATH"]
    if os.getenv("NARROW_AI_EXCLUSIVE_HEAVY_JOBS"):
        env_overlay["model"]["exclusive_heavy_jobs"] = os.environ[
            "NARROW_AI_EXCLUSIVE_HEAVY_JOBS"
        ].lower() in {"1", "true", "yes"}
    if os.getenv("NARROW_AI_CACHE_MODE"):
        env_overlay["model"]["cache_mode"] = os.environ["NARROW_AI_CACHE_MODE"]
    if os.getenv("NARROW_AI_LOG_LEVEL"):
        env_overlay.setdefault("logging", {})["level"] = os.environ["NARROW_AI_LOG_LEVEL"]
    merged = _deep_merge(data, {k: v for k, v in env_overlay.items() if v})
    return Settings(**merged)


def repo_root() -> Path:
    return REPO_ROOT

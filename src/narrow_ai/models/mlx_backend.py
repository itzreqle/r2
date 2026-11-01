"""MLX-LM inference backend. Downloads only when explicitly loaded."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from narrow_ai.models.base import GenerationResult, ModelBackend

logger = logging.getLogger(__name__)


class MlxBackend(ModelBackend):
    name = "mlx"

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: Path | None = None,
        max_tokens: int = 256,
        temperature: float = 0.1,
        max_kv_size: int = 4096,
        load: bool = True,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_kv_size = max_kv_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_time_ms = 0.0
        if load:
            self.load()

    def load(self) -> None:
        try:
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Run: uv sync  (ml dependency group)"
            ) from exc
        t0 = time.perf_counter()
        adapter = str(self.adapter_path) if self.adapter_path else None
        logger.info("Loading MLX model %s (adapter=%s)", self.model_id, adapter)
        if adapter:
            self._model, self._tokenizer = mlx_load(self.model_id, adapter_path=adapter)
        else:
            self._model, self._tokenizer = mlx_load(self.model_id)
        self._load_time_ms = (time.perf_counter() - t0) * 1000
        logger.info("Model loaded in %.0f ms", self._load_time_ms)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            self.load()
        from mlx_lm import generate as mlx_generate

        tok = self._tokenizer
        if getattr(tok, "chat_template", None) is not None:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        mt = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        t0 = time.perf_counter()
        # mlx_lm.generate API: temp via sampler in newer versions; keep kwargs minimal
        kwargs: dict[str, Any] = {"max_tokens": mt, "verbose": False}
        try:
            text = mlx_generate(self._model, tok, prompt=prompt, **kwargs)
        except TypeError:
            text = mlx_generate(self._model, tok, prompt=prompt, max_tokens=mt)
        elapsed = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            text=text if isinstance(text, str) else str(text),
            load_time_ms=self._load_time_ms,
            generate_time_ms=elapsed,
            backend=self.name,
            meta={"model_id": self.model_id, "temperature": temp, "max_kv_size": self.max_kv_size},
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None

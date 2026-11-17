"""Ollama (llama.cpp) backend, used as a same-model baseline for the MLX backend.

The prompt is rendered with the *same* Hugging Face tokenizer and chat template as the
MLX backend (``enable_thinking=False``) and sent with ``raw=True``, so both runtimes see
identical text. Ollama keeps the KV state of the last prompt in memory and reuses the
longest matching prefix, but nothing survives an unload: that is the behaviour we compare.

* No constraint: the JSON prefill is appended to the raw prompt, as in the MLX backend.
* ``constraint_schema``: no prefill; the schema is passed as Ollama's ``format`` and compiled
  to a llama.cpp grammar, so the whole decision object is constrained.
* ``constrain_intents`` (intent-name mask) is not supported and is ignored.

Talks only to a loopback Ollama server; never pulls models.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from narrow_ai.models.base import GenerationResult, ModelBackend
from narrow_ai.models.hub import local_or_repo

LOOPBACK = ("http://127.0.0.1:", "http://localhost:")


class OllamaBackend(ModelBackend):
    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        *,
        tokenizer_id: str = "mlx-community/Qwen3-1.7B-4bit",
        host: str = "http://127.0.0.1:11434",
        max_tokens: int = 128,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        tokenizer: Any = None,
        opener: Any = None,
    ) -> None:
        if not host.startswith(LOOPBACK):
            raise ValueError("OllamaBackend only talks to a loopback server")
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._open = opener or urllib.request.urlopen
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(local_or_repo(tokenizer_id))
        self._tokenizer = tokenizer

    def render(self, messages: list[dict[str, str]], prefill: str = "") -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt + prefill

    def _post(self, path: str, body: dict) -> Any:
        req = urllib.request.Request(
            self.host + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        return self._open(req)

    def unload(self) -> None:
        """Evict the model (and its in-memory KV state), as a process restart would."""
        self._post("/api/generate", {"model": self.model, "keep_alive": 0}).read()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prefill: str = "",
        constrain_intents: list[str] | None = None,
        constraint_schema: dict | None = None,
        conversation: bool = False,
    ) -> GenerationResult:
        _ = constrain_intents, conversation
        lead = "" if constraint_schema else prefill
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": self.render(messages, lead),
            "raw": True,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.0 if temperature is None else temperature,
                "num_predict": max_tokens or self.max_tokens,
                "num_ctx": self.num_ctx,
                "seed": 0,
                "stop": ["<|im_end|>"],
            },
        }
        if constraint_schema:
            body["format"] = constraint_schema
        t0 = time.perf_counter()
        ttft = 0.0
        out, last = "", {}
        with self._post("/api/generate", body) as resp:
            for line in resp:
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev.get("error"):
                    raise RuntimeError(f"ollama: {ev['error']}")
                piece = ev.get("response", "")
                if piece and not ttft:
                    ttft = (time.perf_counter() - t0) * 1000
                out += piece
                last = ev
        total = (time.perf_counter() - t0) * 1000
        eval_ns = last.get("eval_duration") or 0
        n_out = last.get("eval_count") or 0
        prompt_ns = last.get("prompt_eval_duration") or 0
        return GenerationResult(
            text=lead + out,
            load_time_ms=(last.get("load_duration") or 0) / 1e6,
            generate_time_ms=total,
            token_count=n_out,
            backend=self.name,
            ttft_ms=ttft or total,
            prompt_tokens=last.get("prompt_eval_count") or 0,
            generation_tps=n_out / (eval_ns / 1e9) if eval_ns else 0.0,
            meta={"prompt_eval_ms": prompt_ns / 1e6, "model": self.model},
            cache_source="ollama",
        )

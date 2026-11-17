"""OllamaBackend request shaping, without a server, model or network."""

from __future__ import annotations

import io
import json

import pytest

from narrow_ai.models.ollama_backend import OllamaBackend


class FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class FakeServer:
    def __init__(self, pieces):
        self.pieces = pieces
        self.requests = []

    def __call__(self, req):
        self.requests.append((req.full_url, json.loads(req.data)))
        lines = [json.dumps({"response": p}) for p in self.pieces]
        lines.append(json.dumps({"response": "", "done": True, "load_duration": 2_000_000,
                                 "prompt_eval_count": 7, "prompt_eval_duration": 5_000_000,
                                 "eval_count": 3, "eval_duration": 30_000_000}))
        return io.BytesIO(("\n".join(lines) + "\n").encode())


MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "git status"}]


def make(pieces):
    server = FakeServer(pieces)
    return OllamaBackend("m", tokenizer=FakeTokenizer(), opener=server), server


def test_free_decoding_appends_prefill_to_raw_prompt():
    be, server = make(['git_status", ', '"parameters": {}}'])
    r = be.generate(MSGS, prefill='{"intent": "')
    url, body = server.requests[0]
    assert url.endswith("/api/generate") and body["raw"] is True
    assert body["prompt"] == '<system>S<user>git status<assistant>{"intent": "'
    assert "format" not in body and body["options"]["temperature"] == 0.0
    assert r.text == '{"intent": "git_status", "parameters": {}}'
    assert r.load_time_ms == pytest.approx(2.0) and r.token_count == 3


def test_schema_mode_drops_prefill_and_sends_format():
    schema = {"type": "object"}
    be, server = make(['{"intent": "git_status", "parameters": {}}'])
    r = be.generate(MSGS, prefill='{"intent": "', constraint_schema=schema)
    _, body = server.requests[0]
    assert body["format"] == schema and body["prompt"].endswith("<assistant>")
    assert r.text.startswith('{"intent": "git_status"')


def test_refuses_non_loopback_host():
    with pytest.raises(ValueError):
        OllamaBackend("m", host="http://example.com:11434", tokenizer=FakeTokenizer())

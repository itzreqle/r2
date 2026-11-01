"""Bounded short-term conversation memory (in-process)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationStore:
    max_turns: int = 8
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        # Keep system + last N user/assistant pairs roughly
        if len(self.messages) > self.max_turns * 2 + 1:
            system = [m for m in self.messages if m.role == "system"][:1]
            rest = [m for m in self.messages if m.role != "system"]
            self.messages = system + rest[-(self.max_turns * 2) :]

    def as_chat(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()

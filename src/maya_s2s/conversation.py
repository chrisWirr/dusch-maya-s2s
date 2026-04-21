from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Turn:
    role: str
    content: str


@dataclass(slots=True)
class ConversationState:
    system_prompt: str | None = None
    speaker_id: int = 0
    turns: list[Turn] = field(default_factory=list)

    def append_user(self, text: str) -> None:
        self.turns.append(Turn(role="user", content=text))

    def append_assistant(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    def trim(self, max_turns: int) -> None:
        if max_turns <= 0:
            self.turns.clear()
            return
        max_items = max_turns * 2
        if len(self.turns) > max_items:
            self.turns = self.turns[-max_items:]

    def as_messages(self, latest_user_text: str, max_turns: int) -> list[dict[str, str]]:
        self.trim(max_turns)
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend({"role": turn.role, "content": turn.content} for turn in self.turns)
        messages.append({"role": "user", "content": latest_user_text})
        return messages


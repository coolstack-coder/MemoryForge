"""The LLM interface every backend implements."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol, Sequence, TypedDict, runtime_checkable


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    backend: str = ""
    simulated: bool = False


class LLMError(RuntimeError):
    pass


class UsageMeter:
    """Thread-safe running total of tokens and calls for one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: Usage) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }


@runtime_checkable
class LLM(Protocol):
    name: str
    is_simulation: bool
    meter: UsageMeter

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion: ...

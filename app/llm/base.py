"""
LLM provider abstraction.

Providers are *only* allowed to do two things: turn a prompt into text, and
report which model answered. They never touch state, files or the DAG. That
keeps the application swappable across OpenCode / Ollama / any OpenAI-compatible
API without the orchestration logic noticing.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """
    Raised when a provider cannot produce a completion.

    `retryable` separates "this will probably still be broken in half a second"
    (a 429 or a dropped connection) from "no amount of waiting will help"
    (the CLI is not installed, the endpoint answered with the wrong shape).
    Both are errors, but only the first is worth spending attempts on.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # convenience so str(response) is the text
        return self.text


class LLMProvider(abc.ABC):
    """Base class for every provider."""

    name: str = "base"

    def __init__(self, model: str = "", timeout: float = 300.0, max_retries: int = 2):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._token_count = 0
        self._total_seconds = 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "calls": self._call_count,
            "tokens": self._token_count,
            "total_seconds": round(self._total_seconds, 2),
        }

    @abc.abstractmethod
    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Provider-specific network/CLI call. Implemented by subclasses."""

    async def complete(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Return a completion, retrying transient failures with backoff."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self._call(prompt, system_prompt), timeout=self.timeout
                )
                response.duration_seconds = round(time.monotonic() - started, 3)
                response.provider = self.name
                response.model = response.model or self.model
                self._call_count += 1
                self._token_count += response.prompt_tokens + response.completion_tokens
                self._total_seconds += response.duration_seconds
                return response
            except asyncio.TimeoutError as exc:
                last_error = LLMError(f"{self.name}: timed out after {self.timeout}s")
                last_error.__cause__ = exc
            except LLMError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
            except Exception as exc:  # transport / parse / CLI failure
                last_error = LLMError(f"{self.name}: {exc}")
                last_error.__cause__ = exc
            if attempt < self.max_retries:
                await asyncio.sleep(min(2**attempt * 0.5, 8.0))
        raise last_error or LLMError(f"{self.name}: unknown failure")

    async def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        """Default chat implementation: flatten to a single prompt."""
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        flat = "\n\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages
            if m.get("role") != "system"
        )
        return await self.complete(flat, system)

    async def complete_text(self, prompt: str, system_prompt: str = "") -> str:
        """Backwards-compatible helper returning just the text."""
        return (await self.complete(prompt, system_prompt)).text

    async def health_check(self) -> bool:
        """Whether this provider is currently usable. Never raises."""
        try:
            await self.complete("ping", "Reply with the single word: ok")
            return True
        except Exception:
            return False

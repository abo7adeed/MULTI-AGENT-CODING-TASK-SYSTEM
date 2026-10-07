"""
OpenCode provider -- wraps the locally installed OpenCode CLI.

The model is never hard-coded: it comes from the constructor, then
`OPENCODE_MODEL`, then the provider default. Call `list_available_models()` to
discover what the local install actually offers rather than trusting a docstring.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

from app.llm.base import LLMError, LLMProvider, LLMResponse


class OpenCodeProvider(LLMProvider):
    name = "opencode"

    DEFAULT_MODEL = "nemotron-3.5-lightning-free"

    def __init__(self, model: str | None = None, binary: str = "opencode", **kwargs):
        resolved = model or os.getenv("OPENCODE_MODEL") or self.DEFAULT_MODEL
        super().__init__(model=resolved, **kwargs)
        self.binary = binary

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    async def _run_cli(self, args: list[str]) -> str:
        if not self.is_available():
            raise LLMError(
                f"'{self.binary}' CLI not found on PATH. "
                "Install OpenCode or select a different LLM provider.",
                retryable=False,
            )
        proc = await asyncio.create_subprocess_exec(
            self.binary,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise LLMError(f"opencode timed out after {self.timeout}s")
        if proc.returncode != 0:
            raise LLMError(
                f"opencode exited {proc.returncode}: {stderr.decode(errors='replace')[:400]}"
            )
        return stdout.decode(errors="replace")

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        raw = await self._run_cli(["run", "--model", self.model, "--print-logs", "--prompt", full])
        text = _strip_ansi(raw).strip()
        if not text:
            raise LLMError("opencode returned an empty response")
        return LLMResponse(text=text, model=self.model, raw={"stdout": raw})

    async def list_available_models(self) -> list[str]:
        """Ask the local install which models it can reach.

        Falls back to the configured model so callers always get something.
        """
        try:
            raw = await self._run_cli(["models"])
        except Exception:
            return [self.model]
        models: list[str] = []
        for line in raw.splitlines():
            candidate = line.strip().lstrip("-*• ").split("/")[-1].strip()
            if candidate and " " not in candidate:
                models.append(candidate)
        return models or [self.model]


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


async def list_opencode_models(binary: str = "opencode") -> list[str]:
    """Module-level helper used by the Settings page."""
    return await OpenCodeProvider(binary=binary).list_available_models()


def parse_json_response(text: str) -> object | None:
    """Best-effort JSON extraction from an LLM reply (handles ``` fences)."""
    cleaned = _strip_ansi(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    cleaned = cleaned.strip()
    # The document must start at whichever bracket comes first: trying "{" first
    # would turn a list of objects such as [{"id": "t1"}] into a single object
    # and quietly lose every element but one.
    candidates = sorted(
        (cleaned.find(opener), opener, closer)
        for opener, closer in (("{", "}"), ("[", "]"))
        if cleaned.find(opener) != -1
    )
    for start, _, closer in candidates:
        end = cleaned.rfind(closer)
        if end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None

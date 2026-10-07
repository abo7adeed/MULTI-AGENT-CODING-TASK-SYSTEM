"""
Mock LLM provider.

Deterministic, offline, and scriptable. The test suite depends on this and
never on a real model, which is the whole point of the provider abstraction.

`ScriptedLLM` lets a test queue exact responses in order; `EchoLLM` echoes the
prompt back (useful for asserting prompt construction).
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict, deque
from typing import Callable

from app.llm.base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Returns a single canned response for every prompt."""

    name = "mock"

    #: Sentinel: distinguish "no response configured" from an explicit one, so
    #: a scripted mock stays scriptable while the zero-config default can still
    #: satisfy the coder contract.
    DEFAULT_RESPONSE = "MOCK RESPONSE"

    def __init__(self, response: str | None = None, model: str = "mock-model", **kwargs):
        super().__init__(model=model, **kwargs)
        self._response = self.DEFAULT_RESPONSE if response is None else response
        self._unconfigured = response is None
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []
        self.fail_times: int = 0  # raise this many times before succeeding

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("mock provider simulated failure")
        await asyncio.sleep(0)
        if self._unconfigured:
            patch = _patch_reply(prompt, system_prompt)
            if patch is not None:
                return LLMResponse(text=patch, model=self.model, raw={"prompt": prompt})
            review_pattern, review_reply = _REVIEW_RULE
            if review_pattern.search(prompt):
                return LLMResponse(text=review_reply, model=self.model, raw={"prompt": prompt})
        return LLMResponse(text=self._response, model=self.model, raw={"prompt": prompt})


class ScriptedLLM(LLMProvider):
    """Pops responses from a queue; falls back to a default when empty."""

    name = "scripted"

    def __init__(self, responses: list[str] | None = None, default: str = "", model: str = "scripted", **kwargs):
        super().__init__(model=model, **kwargs)
        self._queue = deque(responses or [])
        self._default = default
        self.prompts: list[str] = []

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        text = self._queue.popleft() if self._queue else self._default
        return LLMResponse(text=text, model=self.model)


class CallbackLLM(LLMProvider):
    """Delegates to a user-supplied async or sync callable."""

    name = "callback"

    def __init__(self, fn: Callable[[str, str], str], model: str = "callback", **kwargs):
        super().__init__(model=model, **kwargs)
        self._fn = fn

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        result = self._fn(prompt, system_prompt)
        if asyncio.iscoroutine(result):
            result = await result
        return LLMResponse(text=str(result), model=self.model)


#: Final review asks for {"findings": [...]}. Answering it keeps the offline
#: default from turning its own canned text into a phantom review finding,
#: which reads like a bug in the report. Deliberately *only* the review shape:
#: returning the decomposer's {"tasks": []} would hand back an empty plan,
#: which is worse than letting the caller's fallback keep the existing one.
_REVIEW_RULE = (re.compile(r'"findings"\s*:\s*\['), '{"findings": []}')


class RuleBasedLLM(LLMProvider):
    """
    Deterministic pseudo-LLM for demos: matches regexes on the prompt.

    Lets the whole pipeline run end-to-end with zero network and zero API key
    while still producing a *task-relevant* DAG, which a fixed stub cannot.

    The default rules are keyed on the JSON shapes the application actually
    asks for, so selecting this provider in the UI yields a real (if bland)
    plan rather than empty output. Subclass and override `RULES` to specialise.
    """

    name = "rule-based"

    RULES: list[tuple[re.Pattern[str], str]] = [
        # Decomposer refinement: the prompt quotes the current plan and asks
        # for a revised {"tasks": [...]} document.
        (
            re.compile(r'"tasks"\s*:\s*\['),
            '{"tasks": []}',
        ),
        # Final review: a diff plus {"findings": [...]}.
        _REVIEW_RULE,
        # Task analysis: {"summary", "steps", "risks", "acceptance_criteria"}.
        (
            re.compile(r'"acceptance_criteria"\s*:\s*\['),
            '{"summary": "Rule-based analysis: no model available.", "steps": [], '
            '"risks": [], "acceptance_criteria": []}',
        ),
        # Repository enrichment: {"architecture", "extension_points", ...}.
        (
            re.compile(r'"architecture"\s*:\s*"'),
            '{"architecture": "Rule-based reading of the measured facts.", '
            '"extension_points": [], "risks": [], "relevant_modules": []}',
        ),
    ]

    def __init__(self, default: str = "", model: str = "rule-based", **kwargs):
        super().__init__(model=model, **kwargs)
        self._default = default
        self._hits: dict[str, int] = defaultdict(int)

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        await asyncio.sleep(0)
        patch = _patch_reply(prompt, system_prompt)
        if patch is not None:
            self._hits["coder"] += 1
            return LLMResponse(text=patch, model=self.model)
        for pattern, response in self.RULES:
            if pattern.search(prompt):
                self._hits[pattern.pattern] += 1
                return LLMResponse(text=response, model=self.model)
        return LLMResponse(text=self._default, model=self.model)


def _patch_reply(prompt: str, system_prompt: str) -> str | None:
    """
    Answer a coding task with a valid `<file>` block, or None if this is not a
    coding task.

    An implementation task that emits no `<file>` block fails permanently, so
    an offline provider advertised in the UI must satisfy that contract or every
    demo run dies at the first non-analysis task. Shared by both offline
    providers: the coder envelope is a stronger signal than any JSON shape, so
    it is resolved before the canned response and before `RULES`.
    """
    title_match = _TASK_HEADING.search(prompt)
    if not title_match or _NO_WRITE_ACCESS in system_prompt:
        return None

    title = title_match.group("title").strip()
    scope = _allowed_prefixes(system_prompt)
    # Only ever *add* a file. `<file path>` carries the complete new content,
    # and the excerpts in a prompt are capped and may be hunks, so rewriting an
    # existing file from them truncates the user's code. An offline provider
    # cannot see the whole file; the honest move is to leave it alone.
    target = f"{scope[0]}{slugify(title)}.md" if scope else f"docs/{slugify(title)}.md"
    return _envelope(
        target,
        f"# {title}\n\nOffline demo output; no model was available.\n",
        title,
    )


_TASK_HEADING = re.compile(r"^## Your task\s*\n\*\*(?P<title>.+?)\*\*", re.MULTILINE)
_SCOPE_LINE = re.compile(r"You may ONLY touch these paths:\s*(?P<paths>[^\n]+)")
_NO_WRITE_ACCESS = "You do not have write access"


def slugify(text: str) -> str:
    # Cap the length: task titles are derived from user requests and can be
    # arbitrarily long, and a filename that exceeds the filesystem limit is
    # rejected by the patch layer rather than written.
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60].strip("-")
    return slug or "task"


def _allowed_prefixes(system_prompt: str) -> list[str]:
    match = _SCOPE_LINE.search(system_prompt)
    if not match:
        return []
    return [p.strip() for p in match.group("paths").split(",") if p.strip()]


def _envelope(path: str, content: str, title: str) -> str:
    indented = "\n".join(f"  {line}" if line.strip() else line for line in content.splitlines())
    return (
        f"<note>Offline provider: applying a deterministic demo edit for "
        f"{title!r}.</note>\n\n"
        f'<file path="{path}">\n{indented}\n</file>\n'
    )

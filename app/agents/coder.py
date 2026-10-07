"""
LLM coding agents.

These are the only place the system lets a model influence the filesystem, and
even here the model only *proposes*: it emits a patch envelope, and this module
parses, validates and applies it inside a single isolated worktree. If the
proposal is malformed, out of scope, or points outside the workspace, nothing
is written and the agent is told why.

Two flavours share the machinery:

  * `LLMCoderAgent`    -- implements a task
  * `LLMReviewAgent`   -- analyses and reports, never writes
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from app.agents.base import AgentContext, BaseAgent
from app.agents.patch import (
    ApplyResult,
    apply_patch,
    parse_patch,
    summarise_prompt_contract,
)
from app.agents.roster import AgentRole, get_role
from app.llm.base import LLMError, LLMProvider
from app.logging_config import get_logger
from app.models.domain import AgentResult, FileChange, Task, TaskStatus

logger = get_logger("app.agents.coder")

MAX_REPAIR_ROUNDS = 2


def build_system_prompt(role: AgentRole, writes_files: bool = True) -> str:
    """Role-specific system prompt. Kept short: long prompts waste every call."""
    parts = [
        f"You are the {role.name} in a multi-agent software engineering system.",
        f"Your specialism: {role.description}",
        "",
        role.guidance,
    ]
    if not writes_files:
        parts += [
            "",
            "You do not have write access for this task. Reply with your analysis in "
            "<note> blocks. Do not emit <file> blocks.",
        ]
    else:
        parts += ["", summarise_prompt_contract(list(role.allowed_prefixes))]
    parts += [
        "",
        "Be decisive and complete. If part of the task is genuinely out of scope, say "
        "so in a <note> rather than leaving it silently undone.",
    ]
    return "\n".join(parts)


def build_user_prompt(context: AgentContext) -> str:
    """Assemble the per-task prompt from the budgeted context."""
    sections: list[str] = []

    if context.original_request:
        sections.append(f"## Original request\n{context.original_request}")

    sections.append(
        f"## Your task\n**{context.task.title}**\n\n{context.task.description or '(no further detail)'}"
    )

    if context.instructions:
        sections.append(f"## How to approach this\n{context.instructions}")

    if context.upstream_summaries:
        joined = "\n".join(f"- {s}" for s in context.upstream_summaries)
        sections.append(f"## What upstream agents already did\n{joined}")

    if context.interfaces:
        joined = "\n".join(f"- {i}" for i in context.interfaces[:20])
        sections.append(f"## Interfaces you must honour\n{joined}")

    if context.conventions:
        joined = "\n".join(f"- {c}" for c in context.conventions)
        sections.append(f"## Repository conventions\n{joined}")

    if context.file_contents:
        excerpts = []
        budget = 18_000
        used = 0
        for rel, text in context.file_contents.items():
            if used >= budget:
                excerpts.append(f"### {rel}\n... [omitted: context budget reached]")
                continue
            body = text[:6000]
            excerpts.append(f"### {rel}\n```\n{body}\n```")
            used += len(body)
        sections.append("## Relevant existing files\n" + "\n\n".join(excerpts))

    if context.test_failures:
        joined = "\n".join(f"- {f}" for f in context.test_failures[:15])
        sections.append(f"## Current test failures\n{joined}")

    if context.previous_error:
        sections.append(
            f"## Your previous attempt failed\n{context.previous_error}\n\n"
            "Diagnose the cause and take a different approach this time."
        )

    if context.attempt > 1:
        sections.append(f"This is attempt {context.attempt}. Do not repeat the same work.")

    return "\n\n".join(sections)


class LLMCoderAgent(BaseAgent):
    """
    A real coding agent backed by an LLMProvider.

    The provider is injected, so swapping OpenCode for Ollama or a mock is a
    constructor argument -- the agent logic never changes.
    """

    def __init__(
        self,
        provider: LLMProvider,
        role: str = "generic",
        writes_files: bool | None = None,
        max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    ):
        self.role_def: AgentRole = get_role(role)
        self.role = self.role_def.key
        self.name = self.role_def.name
        self.description = self.role_def.description
        self.capabilities = self.role_def.capabilities
        self.writes_files = (
            self.role_def.writes_files if writes_files is None else writes_files
        )
        self.provider = provider
        self.max_repair_rounds = max_repair_rounds

    # ── execution ───────────────────────────────────────────────────────────

    async def execute(self, task: Task, context: AgentContext) -> AgentResult:
        started = time.time()
        preflight = await self._preflight(context)
        if preflight is not None:
            return preflight

        system_prompt = build_system_prompt(self.role_def, self.writes_files)
        user_prompt = build_user_prompt(context)

        try:
            response = await self.provider.complete(user_prompt, system_prompt)
        except LLMError as exc:
            return self._failure(
                task,
                f"{self.role_def.name} could not reach the model: {exc}",
                [str(exc)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM call failed", extra={"role": self.role})
            return self._failure(task, "unexpected provider error", [str(exc)])

        return await self._interpret(task, context, response.text, started)

    async def _interpret(
        self,
        task: Task,
        context: AgentContext,
        text: str,
        started: float,
    ) -> AgentResult:
        """Parse the reply, apply it, and repair the proposal if it was unusable."""
        # A coder's scope must hold even when the caller did not restate it.
        # Trusting a missing `allowed_prefixes` here would let a backend agent
        # write into the frontend tree purely by omitting a field.
        scope = list(context.allowed_prefixes) or list(self.role_def.allowed_prefixes)

        if not self.writes_files:
            patch = parse_patch(text)
            return self._report(task, patch, started, [], 0, outcome=None, notes_only=True)

        last_errors: list[str] = []
        current = text
        for round_index in range(self.max_repair_rounds + 1):
            patch = parse_patch(current)

            if patch.is_empty:
                if last_errors:
                    break
                return self._failure(
                    task,
                    f"{self.role_def.name} returned no file changes",
                    [
                        "the reply contained no <file> block; "
                        "an implementation task must produce at least one"
                    ],
                )

            outcome = apply_patch(
                patch,
                workspace=_as_path(context.workspace),
                allowed_prefixes=scope or None,
            )
            if outcome.written or outcome.deleted:
                return self._report(
                    task, patch, started, last_errors, round_index, outcome=outcome
                )

            last_errors = [r["reason"] for r in outcome.rejected]
            if round_index >= self.max_repair_rounds:
                break
            logger.info(
                "Patch rejected, asking for a correction",
                extra={"role": self.role, "reasons": last_errors},
            )
            current = await self._repair(task, context, current, last_errors, scope)

        return self._failure(
            task,
            f"{self.role_def.name} produced no applicable change after "
            f"{self.max_repair_rounds} repair attempt(s)",
            last_errors or ["no applicable <file> block found in the model reply"],
        )

    async def _repair(
        self,
        task: Task,
        context: AgentContext,
        previous: str,
        reasons: list[str],
        scope: list[str],
    ) -> str:
        """One focused retry: tell the model exactly what was wrong."""
        reason_lines = "\n".join(f"- {r}" for r in reasons[:8])
        prompt = (
            f"Your previous response for '{task.title}' could not be applied.\n\n"
            f"Problems:\n{reason_lines}\n\n"
            f"Reply again with corrected <file> blocks. Use paths relative to the "
            f"repository root"
            + (f", restricted to: {', '.join(scope)}" if scope else "")
            + ".\n\nPrevious response (for reference):\n"
            + previous[:4000]
        )
        try:
            response = await self.provider.complete(
                prompt, build_system_prompt(self.role_def, self.writes_files)
            )
            return response.text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Repair call failed", extra={"error": str(exc)})
            return previous

    # ── result shaping ──────────────────────────────────────────────────────

    def _report(
        self,
        task: Task,
        patch,
        started: float,
        repair_errors: list[str],
        round_index: int,
        outcome: Optional[ApplyResult] = None,
        notes_only: bool = False,
    ) -> AgentResult:
        """
        Build a SUCCESS result.

        `files_changed` is taken from what was actually written, never from
        what the model proposed -- a read-only agent must not claim files it
        never touched, or the integrator will look for a branch that is empty.
        """
        files = outcome.files_changed if outcome else []
        duration = round(time.time() - started, 3)
        notes = "\n".join(patch.notes)

        if not notes:
            notes = patch.raw.strip()[:1200] if notes_only else ""
        summary = notes or f"{self.role_def.name} completed '{task.title}'."
        if repair_errors:
            summary = f"After {round_index} correction(s): {summary}"

        return self._result(
            task,
            TaskStatus.SUCCESS,
            summary,
            files_changed=files,
            file_changes=[_as_change(p) for p in files],
            duration_seconds=duration,
            errors=repair_errors,
            output={
                "model": self.provider.model,
                "provider": self.provider.name,
                "rounds": round_index + 1,
                "notes": patch.notes,
                "read_only": notes_only,
                "ignored_file_blocks": patch.paths() if notes_only else [],
            },
        )

    async def _preflight(self, context: AgentContext) -> Optional[AgentResult]:
        """Refuse obviously-unrunnable work before spending a model call."""
        if not self.writes_files:
            return None
        from pathlib import Path

        workspace = Path(context.workspace)
        if not workspace.exists():
            return self._failure(
                context.task,
                "workspace does not exist",
                [f"missing workspace: {context.workspace}"],
            )
        if not any(workspace.iterdir()):
            return self._failure(
                context.task,
                "workspace is empty (the base commit was not checked out)",
                [f"empty workspace: {context.workspace}"],
            )
        return None


class LLMReviewAgent(LLMCoderAgent):
    """
    A read-only agent: analysis, planning, review.

    Structurally identical to `LLMCoderAgent` with writes disabled, kept as a
    named subclass so the registry and the UI can tell them apart.
    """

    def __init__(self, provider: LLMProvider, role: str = "code_review", **kwargs):
        super().__init__(provider=provider, role=role, writes_files=False, **kwargs)

    async def _repair(self, task, context, previous, reasons, scope):  # noqa: D102
        return previous  # a read-only agent has nothing to repair


class LLMConflictResolverAgent(LLMCoderAgent):
    """Reads conflict markers and emits a merged file."""

    def __init__(self, provider: LLMProvider, **kwargs):
        super().__init__(provider=provider, role="integration", **kwargs)


def _as_path(value: str):
    from pathlib import Path

    return Path(value)


def _as_change(path: str) -> FileChange:
    return FileChange(path=path, action="modified")


async def run_agent_safely(agent: BaseAgent, task: Task, context: AgentContext) -> AgentResult:
    """
    Run an agent, converting an escaped exception into a FAILED result.

    An agent that raises must not take down the scheduler: the retry policy
    needs a well-formed failure to work with.
    """
    try:
        return await agent.execute(task, context)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent raised", extra={"role": getattr(agent, "role", "?")})
        return AgentResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            summary=f"Agent {getattr(agent, 'role', '?')} raised an exception",
            agent_role=getattr(agent, "role", None),
            errors=[f"{type(exc).__name__}: {exc}"],
        )

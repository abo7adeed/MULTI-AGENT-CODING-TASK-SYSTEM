"""
Mock agents.

Deterministic stand-ins for every real agent. The whole test suite runs on
these, which is why it never touches a network or needs an API key.

Crucially these are not no-ops: a MockAgent writes real files into its
workspace and produces a real diff, so the integrator, the conflict resolver
and the test runner all exercise genuine code paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

from app.agents.base import AgentContext, BaseAgent
from app.agents.patch import ApplyResult, apply_patch, parse_patch
from app.agents.roster import AgentRole, get_role
from app.models.domain import AgentResult, FileChange, Task, TaskStatus


class MockAgent(BaseAgent):
    """
    A configurable, deterministic agent.

    Args:
        succeed:        whether tasks complete successfully
        fail_task_ids:  specific task ids that must fail (for propagation tests)
        fail_times:     fail the first N attempts of every task, then succeed
        write_files:    actually write files into the workspace
        content:        override the generated file body
        delay:          artificial latency, to make parallelism observable
    """

    def __init__(
        self,
        role: str = "generic",
        succeed: bool = True,
        fail_task_ids: Iterable[str] | None = None,
        fail_times: int = 0,
        write_files: bool | None = None,
        content: str | None = None,
        delay: float = 0.01,
        commit: bool = False,
    ):
        self.role_def: AgentRole = get_role(role)
        self.role = self.role_def.key
        self.name = f"Mock {self.role_def.name}"
        self.capabilities = self.role_def.capabilities
        self.description = f"Deterministic mock of the {self.role_def.name}."
        # Default to the role's own capability: a read-only role like `planner`
        # must never appear to have written files.
        self.writes_files = self.role_def.writes_files if write_files is None else write_files
        self.succeed = succeed
        self.fail_task_ids = set(fail_task_ids or ())
        self.fail_times = fail_times
        self._attempts: dict[str, int] = {}
        self.content = content
        self.delay = delay
        self.commit = commit
        #: Populated for assertions in tests.
        self.calls: list[str] = []

    async def execute(self, task: Task, context: AgentContext) -> AgentResult:
        self.calls.append(task.id)
        if self.delay:
            await asyncio.sleep(self.delay)

        self._attempts[task.id] = self._attempts.get(task.id, 0) + 1
        attempt = self._attempts[task.id]

        should_fail = (
            not self.succeed
            or task.id in self.fail_task_ids
            or attempt <= self.fail_times
        )
        if should_fail:
            reason = (
                f"mock: forced failure for {task.id} (attempt {attempt})"
                if (not self.succeed or task.id in self.fail_task_ids)
                else f"mock: transient failure on attempt {attempt}"
            )
            return self._failure(task, reason, [reason])

        if not self.writes_files:
            return self._result(
                task,
                TaskStatus.SUCCESS,
                f"Mock {self.role_def.name} analysed '{task.title}' and produced a plan.",
                recommendations=[f"Consider splitting '{task.title}' further"],
            )

        written = self._write(task, context)
        return self._result(
            task,
            TaskStatus.SUCCESS,
            f"Mock {self.role_def.name} implemented '{task.title}' "
            f"({len(written)} file(s) written).",
            files_changed=written,
            file_changes=[
                FileChange(path=p, action="created") for p in written
            ],
            tests_passed=1 if task.type.value == "testing" else 0,
            output={"mock": True, "attempt": attempt},
        )

    # ── file writing ────────────────────────────────────────────────────────

    def _write(self, task: Task, context: AgentContext) -> list[str]:
        target_dir = self._target_dir(task, context)
        path = Path(context.workspace) / target_dir / f"{self.role}_task.py"
        body = self.content or self._body(task, context)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            return []
        return [str(path.relative_to(context.workspace)).replace("\\", "/")]

    def _target_dir(self, task: Task, context: AgentContext) -> str:
        allowed = [p for p in context.allowed_prefixes if not p.endswith("*")]
        for prefix in allowed:
            candidate = (Path(context.workspace) / prefix.rstrip("/")).resolve()
            try:
                candidate.relative_to(Path(context.workspace).resolve())
                return prefix.rstrip("/")
            except ValueError:
                continue
        return "generated"

    def _body(self, task: Task, context: AgentContext) -> str:
        return (
            f'"""Generated by the mock {self.role_def.name} for task {task.id}."""\n\n'
            f"from __future__ import annotations\n\n"
            f'TASK_TITLE = {task.title!r}\n'
            f"TASK_TYPE = {task.type.value!r}\n\n\n"
            f"def run() -> str:\n"
            f'    """Entry point for the {task.title} work."""\n'
            f"    return TASK_TITLE\n"
        )


class ScriptedMockAgent(MockAgent):
    """Mock agent that writes exact content supplied by the test."""

    def __init__(self, content: str, role: str = "generic", **kwargs):
        super().__init__(role=role, content=content, **kwargs)

    def _write(self, task: Task, context: AgentContext) -> list[str]:
        rel = f"generated/{task.id}.txt"
        path = Path(context.workspace) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content or "", encoding="utf-8")
        return [rel]


class FailingAgent(BaseAgent):
    """Always fails. Used to test failure propagation deterministically."""

    def __init__(self, reason: str = "injected failure", role: str = "generic", fail_times: int = 0):
        self.role = role
        self.name = f"Failing {role} agent"
        self.reason = reason
        self.writes_files = False
        self._attempts = 0
        self._fail_times = fail_times

    async def execute(self, task: Task, context: AgentContext) -> AgentResult:
        await asyncio.sleep(0)
        self._attempts += 1
        if self._fail_times and self._attempts > self._fail_times:
            return self._result(task, TaskStatus.SUCCESS, "recovered")
        return self._failure(task, f"{self.reason} (attempt {self._attempts})", [self.reason])


class ConflictMockAgent(BaseAgent):
    """
    Writes to a caller-chosen path, so two instances can deliberately collide
    and exercise the real conflict-resolution path.
    """

    def __init__(self, rel_path: str, content: str, role: str = "generic"):
        self.role = role
        self.name = f"Conflicting {role} agent"
        self.rel_path = rel_path
        self.content = content
        self.writes_files = True

    async def execute(self, task: Task, context: AgentContext) -> AgentResult:
        await asyncio.sleep(0)
        path = Path(context.workspace) / self.rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content.format(task=task.title), encoding="utf-8")
        return self._result(
            task,
            TaskStatus.SUCCESS,
            f"wrote {self.rel_path}",
            files_changed=[self.rel_path],
        )


def build_mock_roster(
    succeed: bool = True,
    fail_task_ids: Iterable[str] | None = None,
    write_files: bool = True,
    delay: float = 0.01,
) -> dict[str, MockAgent]:
    """One MockAgent per role, for wiring a complete test registry."""
    from app.agents.roster import ROSTER

    return {
        role.key: MockAgent(
            role=role.key,
            succeed=succeed,
            fail_task_ids=fail_task_ids,
            write_files=write_files,
            delay=delay,
        )
        for role in ROSTER
    }


def apply_text_patch(text: str, workspace: str) -> ApplyResult:
    """Convenience wrapper used by tests to assert patch behaviour."""
    return apply_patch(parse_patch(text), Path(workspace))

"""
AgentExecutor.

The seam between the scheduler and the agents. One `run_task` call is what the
scheduler's runner actually invokes, and it is responsible for the whole
per-task pipeline:

    build context -> create isolated workspace -> run agent -> commit -> clean up

Workspace isolation is not optional here. Two agents must never share a
directory, so every task gets its own git worktree, and it is removed (or
preserved on failure, for debugging) when the task ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Optional

from app.agents.base import AgentContext, BaseAgent
from app.agents.coder import run_agent_safely
from app.agents.registry import AgentRegistry
from app.agents.selector import AgentSelector
from app.brain.context import ContextManager, RepositorySnapshot
from app.events import EventBus, EventType, get_event_bus
from app.git.manager import GitManager
from app.logging_config import get_logger
from app.models.domain import AgentResult, ExecutionState, Task, TaskStatus

logger = get_logger("app.agents.executor")


class AgentExecutor:
    """
    Executes one task with one agent, in one isolated workspace.

    The executor raises on failure so the scheduler's retry policy can see it;
    a well-formed `AgentResult` is still recorded first, so the UI can show why.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        selector: AgentSelector,
        state: ExecutionState,
        context_manager: Optional[ContextManager] = None,
        git_manager: Optional[GitManager] = None,
        workspace_root: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        orchestration_id: str = "",
        base_branch: str = "main",
        keep_failed_workspaces: bool = True,
    ):
        self.registry = registry
        self.selector = selector
        self.state = state
        self.context_manager = context_manager or ContextManager(
            snapshot=RepositorySnapshot(root=state.repository)
        )
        self.git = git_manager
        self.workspace_root = workspace_root
        self.event_bus = event_bus or get_event_bus()
        self.orchestration_id = orchestration_id
        self.base_branch = base_branch
        self.keep_failed_workspaces = keep_failed_workspaces

    # ── main entry point ────────────────────────────────────────────────────

    async def run_task(self, task: Task, attempt: int = 1) -> AgentResult:
        selection = self.selector.select(task)
        agent = self.registry.find(selection.role)
        if agent is None:
            # A missing agent is a configuration error, not a task failure --
            # but it must be visible, so it is recorded and re-raised.
            message = (
                f"No agent registered for role '{selection.role}' "
                f"(selected because: {selection.reason})"
            )
            logger.error(message, extra={"task_title": task.title})
            result = AgentResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                summary=message,
                agent_role=selection.role,
                errors=[message],
            )
            self._record(result)
            raise RuntimeError(message)

        task.assigned_agent = selection.role
        started = time.time()
        await self._emit(
            EventType.AGENT_STARTED,
            task_id=task.id,
            title=task.title,
            role=selection.role,
            reason=selection.reason,
        )

        workspace, branch = await self._prepare_workspace(task)
        task.workspace = str(workspace) if workspace else None
        task.branch = branch

        context = self._build_context(task, selection.role_def, workspace, attempt)
        result = await run_agent_safely(agent, task, context)

        result.duration_seconds = round(time.time() - started, 3)
        result.attempt = attempt
        result.branch = branch
        result.agent_role = result.agent_role or selection.role

        if result.status == TaskStatus.SUCCESS and branch and self.git is not None:
            commit = await self._commit(workspace, task)
            if commit:
                result.commit = commit

        if workspace is not None:
            await self._cleanup_workspace(workspace, branch, result.status)

        # Mirror the agent's structured output onto the task for the UI.
        task.output.update(result.output)
        task.output["files_changed"] = result.files_changed
        task.output["summary"] = result.summary
        if result.errors:
            task.errors.extend(result.errors)

        self._record(result)
        await self._emit(
            EventType.AGENT_FINISHED,
            task_id=task.id,
            title=task.title,
            role=selection.role,
            status=result.status.value,
            files_changed=result.files_changed,
            duration_seconds=result.duration_seconds,
            summary=result.summary[:400],
        )

        if result.status != TaskStatus.SUCCESS:
            raise TaskExecutionError(result, task)
        return result

    # ── context ─────────────────────────────────────────────────────────────

    def _build_context(
        self,
        task: Task,
        role_def,
        workspace: Path | None,
        attempt: int,
    ) -> AgentContext:
        previous_error = task.errors[-1] if task.errors else ""
        test_failures = self.state.test_results.get("failures", []) if self.state.test_results else []
        return self.context_manager.build(
            task=task,
            workspace=str(workspace) if workspace else self.state.repository,
            repository=self.state.repository,
            original_request=self.state.original_task,
            role_def=role_def,
            dependency_results={
                dep: self.state.agent_results[dep]
                for dep in task.dependencies
                if dep in self.state.agent_results
            },
            agent_results=self.state.agent_results,
            repository_analysis=self.state.repository_analysis,
            test_failures=test_failures,
            previous_error=previous_error,
            previous_output=task.output,
            attempt=attempt,
        )

    # ── workspace isolation ─────────────────────────────────────────────────

    async def _prepare_workspace(self, task: Task) -> tuple[Path | None, str | None]:
        """
        Create a dedicated worktree for this task.

        Without git (or without a configured root) we fall back to running in
        the main tree, which is logged loudly because it defeats isolation.
        """
        if self.git is None or not self.workspace_root:
            return None, None
        branch = f"agent/{_slug(task.title)}-{task.id[:8]}"
        root = Path(self.workspace_root) / task.id
        try:
            if root.exists():
                await self.git.remove_worktree(root)
            await self.git.create_worktree(branch, root, base_point=self.base_branch)
            return root.resolve(), branch
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not create worktree, running in the main tree",
                extra={"task_title": task.title, "error": str(exc)[:300]},
            )
            return None, None

    async def _commit(self, workspace: Path, task: Task) -> str | None:
        if self.git is None:
            return None
        try:
            return await self.git.commit_all(
                workspace,
                f"[{task.type.value}] {task.title}\n\nTask: {task.id}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Commit failed", extra={"error": str(exc)[:300]})
            return None

    async def _cleanup_workspace(
        self, workspace: Path, branch: str | None, status: TaskStatus
    ) -> None:
        if self.git is None:
            return
        # A successful task keeps its branch (the integrator needs the commit)
        # but drops the directory, which is pure disk.
        if self.keep_failed_workspaces and status != TaskStatus.SUCCESS:
            logger.info(
                "Keeping failed workspace for inspection",
                extra={"workspace": str(workspace)},
            )
            return
        with contextlib.suppress(Exception):
            await self.git.remove_worktree(workspace)

    # ── bookkeeping ─────────────────────────────────────────────────────────

    def _record(self, result: AgentResult) -> None:
        self.state.agent_results[result.task_id] = result
        self.state.touch()

    async def _emit(self, event_type: EventType, **data) -> None:
        if not self.orchestration_id:
            return
        with contextlib.suppress(Exception):
            await self.event_bus.publish(self.orchestration_id, event_type, **data)


class TaskExecutionError(RuntimeError):
    """
    Raised when an agent reports failure.

    Carries the structured result so the scheduler can record precise errors
    rather than a flattened string.
    """

    def __init__(self, result: AgentResult, task: Task):
        self.result = result
        self.task = task
        reason = "; ".join(result.errors) or result.summary or "unknown error"
        super().__init__(f"Task '{task.title}' failed: {reason}")


def _slug(text: str) -> str:
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (cleaned[:40] or "task").rstrip("-")


def build_default_registry(provider=None) -> AgentRegistry:
    """
    Registry with one real agent per role when a provider is supplied,
    otherwise a full mock roster. This is what the app boots with.
    """
    from app.agents.coder import LLMCoderAgent
    from app.agents.mock import MockAgent
    from app.agents.roster import ROSTER

    registry = AgentRegistry()
    for role in ROSTER:
        if provider is None:
            registry.register(role.key, MockAgent(role=role.key))
        else:
            registry.register(role.key, LLMCoderAgent(provider=provider, role=role.key))
    return registry

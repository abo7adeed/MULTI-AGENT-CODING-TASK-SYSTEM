"""
Agent interface.

An agent is a pure function from (Task, AgentContext) to AgentResult. It is
given everything it needs up front, it may only write inside its own workspace,
and it must report honestly -- including when it did nothing.

That last point is why `AgentResult` is a first-class model rather than a
string: the orchestrator, the integrator and the UI all make decisions based
on what an agent claims to have done, and those claims must be checkable.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Optional

from app.models.agent_context import AgentContext
from app.models.domain import AgentResult, Task, TaskStatus




class BaseAgent(abc.ABC):
    """Base class for every agent implementation."""

    role: str = "generic"
    name: str = "Base Agent"
    capabilities: tuple[str, ...] = ()
    description: str = ""
    writes_files: bool = True

    @abc.abstractmethod
    async def execute(self, task: Task, context: AgentContext) -> AgentResult:
        """Do the work and report the result. Must not raise for task failure."""

    # ── helpers for subclasses ──────────────────────────────────────────────

    def _result(
        self,
        task: Task,
        status: TaskStatus,
        summary: str,
        **kwargs: Any,
    ) -> AgentResult:
        return AgentResult(
            task_id=task.id,
            status=status,
            summary=summary[:2000],
            agent_role=self.role,
            **kwargs,
        )

    def _failure(self, task: Task, summary: str, errors: list[str]) -> AgentResult:
        return self._result(
            task,
            TaskStatus.FAILED,
            summary,
            errors=errors[:10],
        )

    async def _preflight(self, context: AgentContext) -> Optional[AgentResult]:
        """
        Optional guard run before the real work.

        Returning a result here aborts the task -- used to reject a task whose
        inputs are unusable (missing workspace, vanished files) rather than
        burning an LLM call on it.
        """
        return None

    @staticmethod
    def _now() -> float:
        return time.time()

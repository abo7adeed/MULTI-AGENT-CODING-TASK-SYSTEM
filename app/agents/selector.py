"""
Agent selector.

Chooses which role takes a task. The decision is deterministic and explainable:
task type first (from the roster), then explicit assignment, then a keyword
fallback. Nothing here calls an LLM -- routing must never depend on a model
being available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.roster import ROSTER, AgentRole, get_role, roles_for
from app.models.domain import Task, TaskType

#: Keyword -> role key. Checked in order against title + description + type.
KEYWORD_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(plan|roadmap|decompose|break down|breakdown)\b", re.I), "planner"),
    (re.compile(r"\b(architect|architecture|module boundar|contract|design)\b", re.I), "architect"),
    (re.compile(r"\b(explor|analys|inspect|survey|audit the repo|understand the repo)\b", re.I), "repository_analyst"),
    (re.compile(r"\b(test|pytest|coverage|spec|fixture)\b", re.I), "testing"),
    (re.compile(r"\b(fix|debug|reproduc|traceback|stack ?trace|regression)\b", re.I), "debugging"),
    (re.compile(r"\b(review|audit the (code|diff)|security)\b", re.I), "security"),
    (re.compile(r"\b(docker|compose|ci\b|cd\b|pipeline|deploy|infra|kubernetes|helm)\b", re.I), "devops"),
    (re.compile(r"\b(migrat|schema|database|postgres|sql|index|orm|table)\b", re.I), "database"),
    (re.compile(r"\b(rag|embedding|vector|llm|model|retriev|agent|ml\b|ai\b|prompt)\b", re.I), "ai_ml"),
    (re.compile(r"\b(react|frontend|ui\b|ux\b|component|css|tailwind|page|screen)\b", re.I), "frontend"),
    (re.compile(r"\b(api|backend|endpoint|service|server|route|middleware|auth)\b", re.I), "backend"),
    (re.compile(r"\b(document|readme|docstring|guide|tutorial)\b", re.I), "documentation"),
    (re.compile(r"\b(refactor|clean ?up|restructure|rename|extract)\b", re.I), "refactor"),
    (re.compile(r"\b(merge|conflict|integrat)\b", re.I), "integration"),
)


@dataclass(frozen=True)
class Selection:
    role: str
    reason: str
    role_def: AgentRole
    alternatives: tuple[str, ...] = ()


class AgentSelector:
    """
    Stateless, deterministic role router.

        selector = AgentSelector()
        selector.select(task).role   # -> "backend"
    """

    def __init__(self, roster: tuple[AgentRole, ...] = ROSTER, default_role: str = "generic"):
        self._roster = roster
        self._default = default_role
        self._by_type: dict[TaskType, list[AgentRole]] = {}
        for role in roster:
            for task_type in role.task_types:
                self._by_type.setdefault(task_type, []).append(role)
        for roles in self._by_type.values():
            roles.sort(key=lambda r: (r.priority, r.key))

    def select(self, task: Task) -> Selection:
        # 1. An explicit assignment always wins.
        if task.assigned_agent:
            key = self._resolve(task.assigned_agent)
            return Selection(
                role=key,
                reason="explicitly assigned on the task",
                role_def=self._safe_role(key),
            )

        # 2. The declared task type. The catch-all GENERIC role is deliberately
        #    skipped here so that an unrecognised type still gets the chance to
        #    match on the task's actual wording before falling back to it.
        candidates = [
            r for r in (self._by_type.get(task.type) or []) if r.key != self._default
        ] or (self._by_type.get(task.type) or [])
        if candidates and candidates[0].key != self._default:
            chosen = candidates[0]
            return Selection(
                role=chosen.key,
                reason=f"task type {task.type.value!r}",
                role_def=chosen,
                alternatives=tuple(r.key for r in candidates[1:]),
            )

        # 3. Keyword fallback over the human-readable text.
        haystack = f"{task.title} {task.description} {task.type.value}"
        for pattern, key in KEYWORD_RULES:
            if pattern.search(haystack):
                return Selection(
                    role=key,
                    reason=f"keyword match /{pattern.pattern[:32]}/",
                    role_def=self._safe_role(key),
                )

        return Selection(
            role=self._default,
            reason="no rule matched; fell back to the general agent",
            role_def=self._safe_role(self._default),
        )

    def select_agent_for_task(self, task: Task) -> str:
        """String-only convenience, kept for the existing call sites."""
        return self.select(task).role

    def roles_for_type(self, task_type: TaskType) -> list[str]:
        return [r.key for r in roles_for(task_type)]

    @staticmethod
    def _resolve(name: str) -> str:
        name = name.strip().lower()
        from app.agents.registry import ALIASES

        return ALIASES.get(name, name)

    def _safe_role(self, key: str) -> AgentRole:
        try:
            return get_role(key)
        except KeyError:
            return get_role("generic")

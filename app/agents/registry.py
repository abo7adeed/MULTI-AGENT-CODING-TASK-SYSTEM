"""
Agent registry.

Maps a role key to a concrete agent instance. Aliases keep older role names
(`backend_agent`, `planner_agent`, ...) working so nothing that already exists
breaks, while the canonical keys stay short (`backend`, `planner`).
"""

from __future__ import annotations

import threading
from typing import Iterable, Mapping, Optional, Union

from app.agents.base import BaseAgent
from app.agents.roster import BY_KEY, AgentRole
from app.logging_config import get_logger
from app.models.domain import Agent

logger = get_logger("app.agents.registry")

#: Legacy `*_agent` names -> canonical role keys.
ALIASES: dict[str, str] = {
    "planner_agent": "planner",
    "repository_analyst_agent": "repository_analyst",
    "architect_agent": "architect",
    "backend_agent": "backend",
    "frontend_agent": "frontend",
    "db_agent": "database",
    "database_agent": "database",
    "ai_ml_agent": "ai_ml",
    "aiml_agent": "ai_ml",
    "ml_agent": "ai_ml",
    "testing_agent": "testing",
    "devops_agent": "devops",
    "debugging_agent": "debugging",
    "code_review_agent": "code_review",
    "review_agent": "code_review",
    "documentation_agent": "documentation",
    "refactor_agent": "refactor",
    "security_agent": "security",
    "integration_agent": "integration",
    "generic_agent": "generic",
}


class AgentRegistry:
    """Thread-safe role -> agent mapping."""

    def __init__(self, agents: Optional[dict[str, BaseAgent]] = None):
        self._agents: dict[str, BaseAgent] = {}
        self._lock = threading.Lock()
        for role, agent in (agents or {}).items():
            self.register(role, agent)

    # ── registration ────────────────────────────────────────────────────────

    def register(self, role: str, agent: BaseAgent) -> None:
        key = self._canonical(role)
        with self._lock:
            self._agents[key] = agent
        logger.debug("Registered agent", extra={"role": key, "agent": type(agent).__name__})

    def register_role(self, role: AgentRole, agent: BaseAgent) -> None:
        self.register(role.key, agent)

    def register_all(self, agents) -> None:
        """
        Register a batch.

        Accepts either an iterable of agents (the role comes from each agent)
        or a `{role: agent}` mapping -- `build_mock_roster()` returns the
        latter, and iterating it directly would register the role *names*
        rather than the agents.
        """
        items = agents.items() if isinstance(agents, dict) else (
            (getattr(agent, "role", "generic"), agent) for agent in agents
        )
        for role, agent in items:
            if agent is not None:
                self.register(role, agent)

    def unregister(self, role: str) -> bool:
        with self._lock:
            return self._agents.pop(self._canonical(role), None) is not None

    # ── lookup ──────────────────────────────────────────────────────────────

    def get(self, role: str) -> BaseAgent:
        key = self._canonical(role)
        with self._lock:
            agent = self._agents.get(key)
        if agent is None:
            raise KeyError(
                f"No agent registered for role {role!r}. "
                f"Registered: {sorted(self._agents) or 'none'}"
            )
        return agent

    def find(self, role: str) -> Optional[BaseAgent]:
        """Like `get`, but returns None instead of raising."""
        try:
            return self.get(role)
        except KeyError:
            return None

    def has(self, role: str) -> bool:
        return self.find(role) is not None

    @property
    def roles(self) -> list[str]:
        with self._lock:
            return sorted(self._agents)

    def describe(self) -> list[Agent]:
        """Registry contents as Agent models, for the API and the UI."""
        with self._lock:
            items = list(self._agents.items())
        described: list[Agent] = []
        for key, agent in items:
            role_def = BY_KEY.get(key)
            if role_def is not None:
                model = role_def.to_agent()
                model.metadata["implementation"] = type(agent).__name__
            else:
                model = Agent(
                    name=getattr(agent, "name", key),
                    role=key,
                    type="coding",
                    description=getattr(agent, "description", ""),
                    capabilities=list(getattr(agent, "capabilities", ())),
                )
                model.metadata["implementation"] = type(agent).__name__
            described.append(model)
        return described

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, role: object) -> bool:
        return isinstance(role, str) and self.has(role)

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _canonical(role: str) -> str:
        role = role.strip().lower()
        role = ALIASES.get(role, role)
        if role.endswith("_agent"):
            role = ALIASES.get(role, role[: -len("_agent")])
        return role

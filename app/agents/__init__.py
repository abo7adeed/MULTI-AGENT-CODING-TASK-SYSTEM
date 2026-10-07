"""Agent layer: interface, roster, routing, patch protocol, implementations."""

from app.agents.base import AgentContext, BaseAgent
from app.agents.coder import (
    LLMCoderAgent,
    LLMConflictResolverAgent,
    LLMReviewAgent,
    build_system_prompt,
    build_user_prompt,
    run_agent_safely,
)
from app.agents.executor import (
    AgentExecutor,
    TaskExecutionError,
    build_default_registry,
)
from app.agents.mock import (
    ConflictMockAgent,
    FailingAgent,
    MockAgent,
    ScriptedMockAgent,
    build_mock_roster,
)
from app.agents.patch import (
    ApplyResult,
    FileWrite,
    ParsedPatch,
    PatchError,
    apply_patch,
    is_protected,
    parse_patch,
    resolve_within,
)
from app.agents.registry import ALIASES, AgentRegistry
from app.agents.roster import BY_KEY, ROSTER, AgentRole, get_role, roles_for
from app.agents.selector import AgentSelector, Selection

__all__ = [
    "ALIASES",
    "AgentContext",
    "AgentExecutor",
    "AgentRegistry",
    "AgentRole",
    "AgentSelector",
    "ApplyResult",
    "BaseAgent",
    "BY_KEY",
    "ConflictMockAgent",
    "FailingAgent",
    "FileWrite",
    "LLMCoderAgent",
    "LLMConflictResolverAgent",
    "LLMReviewAgent",
    "MockAgent",
    "ParsedPatch",
    "PatchError",
    "ROSTER",
    "ScriptedMockAgent",
    "Selection",
    "TaskExecutionError",
    "apply_patch",
    "build_default_registry",
    "build_mock_roster",
    "build_system_prompt",
    "build_user_prompt",
    "get_role",
    "is_protected",
    "parse_patch",
    "resolve_within",
    "roles_for",
    "run_agent_safely",
]

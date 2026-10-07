"""
Agent context contract.

The explicit surface an agent is allowed to see. It lives in `models` rather
than in `agents` so that `brain` (which assembles it) and `agents` (which
consumes it) can both depend on it without a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    from app.models.domain import Task


@dataclass
class AgentContext:
    """
    Everything an agent is allowed to see.

    Assembled by the ContextManager so the rest of the system never decides
    what "relevant" means ad hoc. Deliberately a small, explicit surface:
    if it is not here, the agent does not get it.
    """

    task: 'Task'
    workspace: str
    repository: str = ""
    original_request: str = ""
    role: str = ""
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    upstream_summaries: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)
    repository_analysis: dict[str, Any] = field(default_factory=dict)
    interfaces: list[str] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    test_failures: list[str] = field(default_factory=list)
    attempt: int = 1
    previous_error: str = ""
    previous_output: dict[str, Any] = field(default_factory=dict)
    allowed_prefixes: list[str] = field(default_factory=list)
    instructions: str = ""
    test_requirements: str = ""

    def as_prompt_dict(self) -> dict[str, Any]:
        """Compact view for prompt construction. Skips empty sections."""
        data: dict[str, Any] = {
            "task_title": self.task.title,
            "task_description": self.task.description,
            "task_type": self.task.type.value,
        }
        optional = {
            "original_request": self.original_request,
            "upstream_summaries": self.upstream_summaries,
            "relevant_files": self.relevant_files,
            "interfaces": self.interfaces,
            "conventions": self.conventions,
            "test_failures": self.test_failures,
            "previous_error": self.previous_error,
            "instructions": self.instructions,
        }
        data.update({k: v for k, v in optional.items() if v})
        return data


__all__ = ["AgentContext"]

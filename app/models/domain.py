"""
Core domain model.

Everything that describes *what* the system is doing lives here. These models
are the contract between the orchestrator, the DAG engine, the agents, the
integrator and the API -- and they are the durable unit persisted to storage.

Deliberate rule: the LLM never mutates these. Models are produced by ordinary
Python from validated LLM output, so the deterministic layer stays authoritative.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def new_id() -> str:
    return str(uuid.uuid4())


# ── Enums ────────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        """No further transition is possible from here."""
        return self in TERMINAL_TASK_STATUSES

    @property
    def is_active(self) -> bool:
        return self in (TaskStatus.RUNNING, TaskStatus.RETRYING)


class TaskType(str, Enum):
    """Canonical task kinds. The decomposer maps free-form LLM output onto these."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    AI_ML = "ai_ml"
    TESTING = "testing"
    DEVOPS = "devops"
    DOCUMENTATION = "documentation"
    INTEGRATION = "integration"
    REVIEW = "review"
    DEBUGGING = "debugging"
    REFACTOR = "refactor"
    SECURITY = "security"
    GENERIC = "generic"


class Phase(str, Enum):
    INITIALIZATION = "INITIALIZATION"
    ANALYSIS = "ANALYSIS"
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    INTEGRATION = "INTEGRATION"
    REVIEW = "REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OrchestrationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrchestrationStatus.COMPLETED,
            OrchestrationStatus.FAILED,
            OrchestrationStatus.CANCELLED,
        )


class FinalStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCESS,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    # BLOCKED is settled for this run: the scheduler will not dispatch it
    # again without an explicit retry, so it must count as terminal here too.
    TaskStatus.BLOCKED,
}


# ── Core models ──────────────────────────────────────────────────────────────


class Task(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    description: str = ""
    type: TaskType = TaskType.GENERIC
    priority: int = 5  # 1 (highest) .. 9 (lowest)
    dependencies: list[str] = Field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    workspace: Optional[str] = None
    branch: Optional[str] = None
    input_context: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

    # observability
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_seconds: float = 0.0
    blocked_reason: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    @field_validator("dependencies")
    @classmethod
    def _no_self_dependency(cls, v: list[str], info):
        tid = info.data.get("id")
        if tid and tid in v:
            raise ValueError(f"Task {tid} cannot depend on itself")
        return v

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: Any) -> Any:
        """Accept free-form strings from the LLM without crashing."""
        if isinstance(v, str):
            normalized = v.strip().lower().replace("-", "_").replace(" ", "_")
            for member in TaskType:
                if member.value == normalized:
                    return member
            aliases = {
                "ml": TaskType.AI_ML,
                "ai": TaskType.AI_ML,
                "llm": TaskType.AI_ML,
                "rag": TaskType.AI_ML,
                "db": TaskType.DATABASE,
                "sql": TaskType.DATABASE,
                "dev_ops": TaskType.DEVOPS,
                "infra": TaskType.DEVOPS,
                "ci_cd": TaskType.DEVOPS,
                "test": TaskType.TESTING,
                "tests": TaskType.TESTING,
                "qa": TaskType.TESTING,
                "ui": TaskType.FRONTEND,
                "web": TaskType.FRONTEND,
                "api": TaskType.BACKEND,
                "service": TaskType.BACKEND,
                "docs": TaskType.DOCUMENTATION,
                "review": TaskType.REVIEW,
                "code_review": TaskType.REVIEW,
                "plan": TaskType.PLANNING,
                "architect": TaskType.ARCHITECTURE,
                "explore": TaskType.ANALYSIS,
            }
            if normalized in aliases:
                return aliases[normalized]
            # Unknown kind: keep it, but normalise so grouping still works.
            return TaskType.GENERIC
        return v


class Agent(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    role: str
    type: str
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    max_concurrency: int = 2
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileChange(BaseModel):
    path: str
    action: str = "modified"  # created | modified | deleted
    additions: int = 0
    deletions: int = 0


class TestOutcome(BaseModel):
    name: str
    passed: bool
    message: str = ""
    duration_seconds: float = 0.0


class AgentResult(BaseModel):
    task_id: str
    status: TaskStatus
    summary: str = ""
    agent_role: Optional[str] = None
    files_changed: list[str] = Field(default_factory=list)
    file_changes: list[FileChange] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    test_outcomes: list[TestOutcome] = Field(default_factory=list)
    commit: Optional[str] = None
    branch: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    attempt: int = 1
    output: dict[str, Any] = Field(default_factory=dict)


class Project(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    description: str = ""
    repository_url: Optional[str] = None
    local_path: str
    base_branch: str = "main"
    default_agent_profile: str = "default"
    created_at: float = Field(default_factory=time.time)

    @field_validator("local_path")
    @classmethod
    def _normalise_path(cls, v: str) -> str:
        import os

        return os.path.abspath(os.path.expanduser(v.strip()))


class DAG(BaseModel):
    tasks: dict[str, Task] = Field(default_factory=dict)
    # Adjacency derived from Task.dependencies: task id -> dependents.
    # Maintained by DAGEngine; never trusted as a source of truth on its own.
    edges: dict[str, list[str]] = Field(default_factory=dict)

    def dependents_of(self, task_id: str) -> list[str]:
        return self.edges.get(task_id, [])

    def add(self, task: Task) -> "DAG":
        self.tasks[task.id] = task
        self.rebuild_edges()
        return self

    def rebuild_edges(self) -> None:
        edges: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in edges:
                    edges[dep].append(task.id)
        self.edges = edges

    @property
    def size(self) -> int:
        return len(self.tasks)


class TestRunResult(BaseModel):
    # Stops pytest trying to collect this model as a test class.
    __test__ = False

    command: str
    passed: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    passed_count: int = 0
    failed_count: int = 0
    duration_seconds: float = 0.0
    skipped: bool = False

    @property
    def summary(self) -> str:
        first = (self.stdout or self.stderr or "").strip().splitlines()
        tail = " | ".join(first[-3:]) if first else ""
        return tail[:500]


class IntegrationReport(BaseModel):
    merged_branches: list[str] = Field(default_factory=list)
    skipped_branches: list[str] = Field(default_factory=list)
    skipped_details: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    conflict_resolutions: list[dict[str, Any]] = Field(default_factory=list)
    test_runs: list[TestRunResult] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    review_findings: list[dict[str, Any]] = Field(default_factory=list)
    report: str = ""
    success: bool = False
    created_at: float = Field(default_factory=time.time)


class ExecutionState(BaseModel):
    project_id: str
    original_task: str
    repository: str
    id: str = Field(default_factory=new_id)
    dag: DAG = Field(default_factory=DAG)
    tasks: dict[str, Task] = Field(default_factory=dict)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    current_phase: Phase = Phase.INITIALIZATION
    errors: list[str] = Field(default_factory=list)
    test_results: dict[str, Any] = Field(default_factory=dict)
    final_status: FinalStatus = FinalStatus.PENDING
    repository_analysis: dict[str, Any] = Field(default_factory=dict)
    task_analysis: dict[str, Any] = Field(default_factory=dict)
    integration_report: Optional[IntegrationReport] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def touch(self) -> None:
        self.updated_at = time.time()

    def record_error(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)

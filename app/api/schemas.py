"""
API schemas.

Separate from the domain models so the wire format can stay stable while the
internals move. Field aliases keep the snake_case the frontend expects.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── projects ─────────────────────────────────────────────────────────────────


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    local_path: str = Field(min_length=1)
    repository_url: Optional[str] = None
    base_branch: str = "main"


class ProjectResponse(BaseModel):
    # Tolerate the extra `default_agent_profile` the route attaches.
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    local_path: str
    repository_url: Optional[str] = None
    base_branch: str = "main"
    created_at: float
    # Convenience aggregates so the dashboard needs one call, not N.
    orchestration_count: int = 0
    running_count: int = 0
    default_agent_profile: str = "default"


# ── tasks ────────────────────────────────────────────────────────────────────


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = ""
    type: str = "generic"
    priority: int = Field(default=5, ge=1, le=9)
    dependencies: list[str] = Field(default_factory=list)
    assigned_agent: Optional[str] = None
    max_retries: int = Field(default=3, ge=0, le=10)


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    priority: Optional[int] = Field(default=None, ge=1, le=9)
    assigned_agent: Optional[str] = None
    status: Optional[str] = None
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)


# ── orchestrations ───────────────────────────────────────────────────────────


class OrchestrationCreate(BaseModel):
    project_id: str
    user_request: str = Field(min_length=3)
    name: Optional[str] = None
    use_llm: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    wait: bool = Field(
        default=False,
        description="Block until the run finishes instead of returning immediately.",
    )



class OrchestrationSummary(BaseModel):
    id: str
    name: str
    status: str
    final_status: str
    current_phase: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    progress: float = 0.0
    duration_seconds: float = 0.0
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class OrchestrationDetail(OrchestrationSummary):
    project_id: str
    original_task: str
    repository: str
    tasks: dict[str, Any] = Field(default_factory=dict)
    agent_results: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    test_results: dict[str, Any] = Field(default_factory=dict)
    repository_analysis: dict[str, Any] = Field(default_factory=dict)
    task_analysis: dict[str, Any] = Field(default_factory=dict)
    integration_report: Optional[dict[str, Any]] = None
    updated_at: float = 0.0


class DAGNode(BaseModel):
    id: str
    title: str
    type: str
    status: str
    priority: int
    agent: Optional[str] = None
    dependencies: list[str] = Field(default_factory=list)
    wave: int = 0
    depth: int = 0
    duration_seconds: float = 0.0
    retry_count: int = 0
    files_changed: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    blocked_reason: Optional[str] = None
    commit: Optional[str] = None


class DAGResponse(BaseModel):
    orchestration_id: str
    total_tasks: int
    waves: list[list[str]] = Field(default_factory=list)
    critical_path: list[str] = Field(default_factory=list)
    max_parallelism: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    nodes: list[DAGNode] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)


# ── agents ───────────────────────────────────────────────────────────────────


class AgentInfo(BaseModel):
    id: Optional[str] = None
    name: str
    role: str
    type: str = "coding"
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    max_concurrency: int = 1
    implementation: Optional[str] = None


class AgentActivity(BaseModel):
    task_id: str
    title: str
    role: Optional[str] = None
    status: str
    summary: str = ""
    files_changed: list[str] = Field(default_factory=list)
    commit: Optional[str] = None
    branch: Optional[str] = None
    duration_seconds: float = 0.0
    attempt: int = 1
    errors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0


class AgentRetryRequest(BaseModel):
    reason: str = ""
    reset_dependencies: bool = Field(
        default=False,
        description="Also reset tasks that were blocked by this failure.",
    )


# ── misc ─────────────────────────────────────────────────────────────────────


class LogEntry(BaseModel):
    sequence: int
    type: str
    timestamp: float
    data: dict[str, Any] = Field(default_factory=dict)


class GitChange(BaseModel):
    branch: Optional[str] = None
    commit: Optional[str] = None
    task_id: str
    title: str
    files_changed: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    provider: str
    model: str
    agents_registered: int
    sandbox: str
    database: str
    uptime_seconds: float
    # From the probe taken at startup: whether the configured provider answered
    # and whether the configured model was among the ones it offered. None means
    # the provider cannot be probed (the offline ones have nothing to reach).
    provider_reachable: Optional[bool] = None
    provider_model_available: Optional[bool] = None
    provider_detail: str = ""


class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None


class SystemConfigUpdate(BaseModel):
    llm_provider: str = Field(description="gemini | api | ollama | opencode | rule-based | mock")
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    persist_to_env: bool = True


class LLMTestRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    prompt: Optional[str] = "Hello! Please reply in one short sentence confirming you are working."


class LLMTestResponse(BaseModel):
    success: bool
    provider: str
    model: str
    latency_ms: float
    reply: str
    error: Optional[str] = None

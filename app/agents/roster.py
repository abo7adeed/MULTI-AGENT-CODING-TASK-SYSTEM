"""
Agent roster.

The ten specialised roles, declared as data rather than as ten hand-written
classes. Each entry says what the role is for, which task types it owns, which
directory it is allowed to write to, and what the orchestrator should tell it.

Data beats subclasses here: adding a role is a dict entry, the selector reads
the same data to route tasks, and the UI reads it again to render the roster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.domain import Agent, TaskType


@dataclass(frozen=True)
class AgentRole:
    key: str
    name: str
    description: str
    task_types: tuple[TaskType, ...]
    capabilities: tuple[str, ...]
    guidance: str
    allowed_prefixes: tuple[str, ...] = ()   # empty = whole workspace
    priority: int = 5                         # lower wins when several match
    max_concurrency: int = 2
    writes_files: bool = True                 # read-only roles set this False

    def to_agent(self) -> Agent:
        return Agent(
            name=self.name,
            role=self.key,
            type="coding",
            description=self.description,
            capabilities=list(self.capabilities),
            max_concurrency=self.max_concurrency,
        )

    def handles(self, task_type: TaskType) -> bool:
        return task_type in self.task_types


PLANNER = AgentRole(
    key="planner",
    name="Planner Agent",
    description="Turns a high-level request into a concrete, ordered implementation plan.",
    task_types=(TaskType.PLANNING,),
    capabilities=("planning", "decomposition", "architecture", "estimation"),
    guidance=(
        "Produce a step-by-step implementation plan. Name the concrete files, modules "
        "and interfaces that must be created. Do not write application code."
    ),
    writes_files=False,
    priority=1,
)

REPOSITORY_ANALYST = AgentRole(
    key="repository_analyst",
    name="Repository Analyst Agent",
    description="Inspects the existing codebase to ground every later decision in reality.",
    task_types=(TaskType.ANALYSIS,),
    capabilities=("repository_analysis", "dependency_mapping", "convention_extraction"),
    guidance=(
        "Describe the existing structure, the frameworks in use, the conventions to "
        "follow and the files most relevant to the task. Do not modify anything."
    ),
    writes_files=False,
    priority=1,
)

ARCHITECT = AgentRole(
    key="architect",
    name="Architecture Agent",
    description="Defines module boundaries, data contracts and integration points.",
    task_types=(TaskType.ARCHITECTURE,),
    capabilities=("architecture", "interface_design", "api_contracts"),
    guidance=(
        "Define module boundaries and the public interfaces between them. Emit type "
        "definitions, schemas and route contracts that the implementation agents will "
        "build against."
    ),
    priority=2,
)

BACKEND = AgentRole(
    key="backend",
    name="Backend Coding Agent",
    description="Implements server-side logic, APIs, services and business rules.",
    task_types=(TaskType.BACKEND,),
    capabilities=("python", "fastapi", "rest", "services", "business_logic"),
    guidance=(
        "Implement production-ready server-side code. Follow the interfaces already "
        "declared upstream, validate all input, and never hard-code secrets."
    ),
    allowed_prefixes=("app/", "src/", "backend/", "server/", "api/", "tests/"),
    priority=3,
)

FRONTEND = AgentRole(
    key="frontend",
    name="Frontend Coding Agent",
    description="Implements the user interface, state management and API integration.",
    task_types=(TaskType.FRONTEND,),
    capabilities=("react", "typescript", "css", "state_management", "accessibility"),
    guidance=(
        "Implement accessible, typed UI components. Keep state in the component tree "
        "or a small store, handle loading and error states explicitly."
    ),
    allowed_prefixes=("frontend/", "src/", "web/", "client/", "app/"),
    priority=3,
)

DATABASE = AgentRole(
    key="database",
    name="Database Agent",
    description="Owns schema, migrations, indexes and data-access layers.",
    task_types=(TaskType.DATABASE,),
    capabilities=("sql", "orm", "migrations", "schema_design", "indexing"),
    guidance=(
        "Write forward-only, reversible migrations. Every migration needs a matching "
        "down path. Never drop or rename a column without an explicit migration step."
    ),
    allowed_prefixes=("migrations/", "alembic/", "app/models/", "src/models/", "db/"),
    priority=2,
)

AI_ML = AgentRole(
    key="ai_ml",
    name="AI/ML Agent",
    description="Implements retrieval, embeddings, model calls and vector search.",
    task_types=(TaskType.AI_ML,),
    capabilities=("llm", "embeddings", "rag", "vector_search", "prompting"),
    guidance=(
        "Implement retrieval and generation pipelines. Batch embedding calls, cache "
        "where sensible, degrade gracefully when the model is unavailable, and never "
        "block the event loop with a synchronous network call."
    ),
    priority=3,
)

TESTING = AgentRole(
    key="testing",
    name="Testing Agent",
    description="Writes and repairs the test suite, and reports exactly what passes.",
    task_types=(TaskType.TESTING,),
    capabilities=("pytest", "integration_tests", "fixtures", "coverage"),
    guidance=(
        "Write deterministic tests: no network, no wall-clock dependence, no ordering "
        "assumptions. Cover the failure paths, not just the happy path."
    ),
    allowed_prefixes=("tests/", "test/", "app/tests/", "spec/", "**/"),
    priority=4,
)

DEVOPS = AgentRole(
    key="devops",
    name="DevOps Agent",
    description="Owns Dockerfiles, compose files, CI pipelines and deployment config.",
    task_types=(TaskType.DEVOPS,),
    capabilities=("docker", "ci_cd", "deployment", "infrastructure"),
    guidance=(
        "Write reproducible builds. Pin image versions, use multi-stage builds, never "
        "bake secrets into an image, and add a healthcheck."
    ),
    allowed_prefixes=("Dockerfile", "docker-compose", ".github/", "deploy/", "infra/", "Makefile"),
    priority=4,
)

DEBUGGING = AgentRole(
    key="debugging",
    name="Debugging Agent",
    description="Diagnoses a failure from evidence and produces the minimal correct fix.",
    task_types=(TaskType.DEBUGGING,),
    capabilities=("debugging", "root_cause_analysis", "regression_fixing"),
    guidance=(
        "You are given a concrete failure. Read the actual traceback, identify the root "
        "cause, and make the smallest change that fixes it. Do not refactor while fixing."
    ),
    priority=1,
)

CODE_REVIEW = AgentRole(
    key="code_review",
    name="Code Review Agent",
    description="Reviews merged changes for correctness, security and regressions.",
    task_types=(TaskType.REVIEW,),
    capabilities=("code_review", "security_review", "quality"),
    guidance=(
        "Review the diff. Report concrete, actionable findings with file and line. "
        "Call out correctness bugs, security holes and missing tests. Do not restate "
        "what the code does."
    ),
    writes_files=False,
    priority=2,
)

DOCUMENTATION = AgentRole(
    key="documentation",
    name="Documentation Agent",
    description="Writes and keeps README, API docs and developer guides accurate.",
    task_types=(TaskType.DOCUMENTATION,),
    capabilities=("documentation", "api_reference", "examples"),
    guidance=(
        "Document what the code actually does. Every example must be runnable as "
        "written. Keep the existing structure and voice of the file you are editing."
    ),
    allowed_prefixes=("docs/", "README", "*.md"),
    priority=6,
)

REFACTOR = AgentRole(
    key="refactor",
    name="Refactoring Agent",
    description="Restructures existing code without changing observable behaviour.",
    task_types=(TaskType.REFACTOR,),
    capabilities=("refactoring", "code_quality", "maintainability"),
    guidance=(
        "Change structure, never behaviour. If a test would fail, you have refactored "
        "too far. Keep the public API stable."
    ),
    priority=5,
)

SECURITY = AgentRole(
    key="security",
    name="Security Agent",
    description="Audits and hardens authentication, authorisation and input handling.",
    task_types=(TaskType.SECURITY,),
    capabilities=("security", "auth", "input_validation", "threat_modeling"),
    guidance=(
        "Find the vulnerability before fixing it. State the threat, then apply the fix. "
        "Reject untrusted input at the boundary; never build a query by concatenation."
    ),
    priority=2,
)

INTEGRATION = AgentRole(
    key="integration",
    name="Integration Agent",
    description="Resolves cross-module conflicts and wires components together.",
    task_types=(TaskType.INTEGRATION,),
    capabilities=("conflict_resolution", "integration", "merge"),
    guidance=(
        "Two agents changed the same region. Understand both intents and produce a "
        "merged result that preserves the behaviour each side was trying to add."
    ),
    priority=1,
)

GENERIC = AgentRole(
    key="generic",
    name="General Coding Agent",
    description="Handles any task that does not match a specialised role.",
    task_types=(TaskType.GENERIC,),
    capabilities=("code_generation", "refactoring", "documentation"),
    guidance="Implement the task precisely. Prefer the smallest change that fully works.",
    priority=9,
)


#: Every role, in roster display order.
ROSTER: tuple[AgentRole, ...] = (
    REPOSITORY_ANALYST,
    PLANNER,
    ARCHITECT,
    BACKEND,
    FRONTEND,
    DATABASE,
    AI_ML,
    TESTING,
    DEVOPS,
    DEBUGGING,
    CODE_REVIEW,
    DOCUMENTATION,
    REFACTOR,
    SECURITY,
    INTEGRATION,
    GENERIC,
)

BY_KEY: dict[str, AgentRole] = {r.key: r for r in ROSTER}


def get_role(key: str) -> AgentRole:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown agent role: {key!r}. Known: {sorted(BY_KEY)}") from None


def roles_for(task_type: TaskType) -> list[AgentRole]:
    """All roles that can take this task type, most specific first."""
    return sorted(
        (r for r in ROSTER if r.handles(task_type)),
        key=lambda r: (r.priority, r.key),
    )

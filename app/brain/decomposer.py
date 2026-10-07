"""
TaskDecomposer.

Turns a `TaskSpecification` into a real DAG. This is the piece that was a hard
coded two-node stub, so it is worth being explicit about how it works:

  1. A deterministic **template catalogue** maps detected deliverables and task
     types onto a dependency graph. This alone produces a sensible, correctly
     ordered, genuinely parallel plan with no model involved.
  2. When a provider is configured, the LLM may **refine** the plan -- but only
     by adding, retitling or re-typing nodes. It cannot invent dependencies that
     create a cycle, because the result is re-validated and repaired.

The LLM therefore improves the plan without ever being trusted with graph
correctness, which is the one job that must stay in Python.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.brain.analyzer import TaskSpecification
from app.engine.dag import CycleDetectedError, DAGEngine
from app.llm.base import LLMProvider
from app.logging_config import get_logger
from app.models.domain import DAG, Task, TaskType, new_id

logger = get_logger("app.brain.decomposer")

PlanBuilder = Callable[[TaskSpecification, dict[str, Any]], DAG]


@dataclass
class Blueprint:
    """One node in the template catalogue, with its dependency rule."""

    key: str
    title: str
    description: str
    task_type: TaskType
    priority: int
    #: Hard dependencies. If one is missing from the active set it is simply
    #: pruned -- the node still runs, just earlier than intended.
    depends_on: tuple[str, ...] = ()
    #: Activation rule: the node exists only if at least one of these is active.
    #: Empty means "always active" (the spine).
    requires_any: tuple[str, ...] = ()
    #: Fan-in: the node additionally depends on every active node in this set.
    #: This is what lets `tests` sit after whichever implementation work exists.
    depends_on_all_of: tuple[str, ...] = ()
    scope: str = ""          # free text steering the agent
    deliverable: str = ""    # which detected deliverable activates it

    def build(self, spec: TaskSpecification) -> Task:
        return Task(
            id=new_id(),
            title=self.title,
            description=f"{self.description}\n\nAcceptance: "
            f"{'; '.join(spec.acceptance_criteria[:3]) or 'the requested behaviour works'}",
            type=self.task_type,
            priority=self.priority,
            input_context={
                "scope": self.scope,
                "blueprint": self.key,
                "deliverables": spec.deliverables,
            },
            tests=[f"tests/test_{self.key}.py"],
        )


#: The catalogue. Order here is the canonical build order.
BLUEPRINTS: tuple[Blueprint, ...] = (
    Blueprint(
        key="analysis",
        title="Analyse the existing repository",
        description=(
            "Inspect the codebase, identify the frameworks in use, the conventions to "
            "follow and the modules the change will touch."
        ),
        task_type=TaskType.ANALYSIS,
        priority=1,
    ),
    Blueprint(
        key="architecture",
        title="Design the architecture and interfaces",
        description=(
            "Define module boundaries and the public interfaces between them. Emit the "
            "type, schema and route contracts the implementation agents will build on."
        ),
        task_type=TaskType.ARCHITECTURE,
        priority=1,
        depends_on=("analysis",),
    ),
    Blueprint(
        key="database",
        title="Design the data layer and migrations",
        description=(
            "Create the schema, write reversible migrations and add the data-access layer."
        ),
        task_type=TaskType.DATABASE,
        priority=2,
        depends_on=("architecture",),
        deliverable="data layer",
    ),
    Blueprint(
        key="ai",
        title="Implement retrieval and generation",
        description=(
            "Implement the embedding, indexing, retrieval and generation pipeline. "
            "Handle the model being unavailable without crashing the request path."
        ),
        task_type=TaskType.AI_ML,
        priority=2,
        depends_on=("architecture",),
        deliverable="retrieval pipeline",
    ),
    Blueprint(
        key="backend",
        title="Implement the backend API",
        description=(
            "Implement the server-side routes, services and business logic behind the "
            "declared interfaces. Validate all input at the boundary."
        ),
        task_type=TaskType.BACKEND,
        priority=3,
        depends_on=("architecture",),
        deliverable="backend API",
    ),
    Blueprint(
        key="auth",
        title="Implement authentication and authorisation",
        description=(
            "Add credential handling, session/token issuance and route-level access "
            "control. Never log credentials or embed secrets in source."
        ),
        task_type=TaskType.SECURITY,
        priority=3,
        depends_on=("architecture",),
        deliverable="authentication",
    ),
    Blueprint(
        key="frontend",
        title="Implement the frontend",
        description=(
            "Build the UI components, wire them to the API client, and handle loading, "
            "empty and error states explicitly."
        ),
        task_type=TaskType.FRONTEND,
        priority=3,
        depends_on=("architecture",),
        deliverable="user interface",
    ),
    Blueprint(
        key="devops",
        title="Containerise and add CI",
        description=(
            "Write the Dockerfile and compose setup, and add the CI pipeline that runs "
            "the test suite. Pin base image versions and add a healthcheck."
        ),
        task_type=TaskType.DEVOPS,
        priority=4,
        requires_any=("container setup", "CI/CD pipeline", "deployment config"),
        # Containerising is a consequence of the code existing, so this fans in
        # from every implementation node rather than from `backend` alone --
        # otherwise a frontend-only request leaves devops as an orphan root.
        depends_on_all_of=("backend", "ai", "database", "auth", "frontend", "generic_work"),
        deliverable="container setup",
    ),
    Blueprint(
        key="ci",
        title="Add the CI/CD pipeline",
        description="Automate build, lint, typecheck and test on every push.",
        task_type=TaskType.DEVOPS,
        priority=5,
        requires_any=("devops",),
        depends_on=("devops",),
        deliverable="CI/CD pipeline",
    ),
    Blueprint(
        key="deployment",
        title="Write deployment configuration",
        description="Produce reproducible deployment config and an operator runbook.",
        task_type=TaskType.DEVOPS,
        priority=6,
        requires_any=("devops",),
        depends_on=("devops",),
        deliverable="deployment config",
    ),
    Blueprint(
        key="tests",
        title="Write the automated test suite",
        description=(
            "Cover the new behaviour with deterministic tests, including the failure "
            "paths. No network access and no wall-clock dependence."
        ),
        task_type=TaskType.TESTING,
        priority=4,
        requires_any=("backend", "auth", "ai", "database", "frontend", "generic_work"),
        depends_on_all_of=("backend", "auth", "ai", "database", "frontend", "generic_work"),
    ),
    Blueprint(
        key="frontend_tests",
        title="Write frontend tests",
        description="Cover component rendering, user interaction and API error states.",
        task_type=TaskType.TESTING,
        priority=5,
        requires_any=("frontend",),
        depends_on=("frontend",),
    ),
    Blueprint(
        key="docs",
        title="Write the documentation",
        description=(
            "Update the README and write an API reference. Every example must be "
            "runnable exactly as written."
        ),
        task_type=TaskType.DOCUMENTATION,
        priority=6,
        requires_any=("backend", "ai", "frontend", "generic_work"),
        depends_on_all_of=("backend", "ai", "frontend", "generic_work"),
        deliverable="documentation",
    ),
    Blueprint(
        key="integration",
        title="Integrate and verify the whole system",
        description=(
            "Wire the modules together, run the full suite and confirm the end-to-end "
            "behaviour matches the acceptance criteria."
        ),
        task_type=TaskType.INTEGRATION,
        priority=5,
        requires_any=("backend", "auth", "ai", "database", "frontend", "tests", "generic_work"),
        depends_on_all_of=("backend", "auth", "ai", "database", "frontend", "tests", "devops", "generic_work"),
    ),
    Blueprint(
        key="review",
        title="Final code review",
        description=(
            "Review the complete diff for correctness, security and missing test "
            "coverage. Report concrete, actionable findings."
        ),
        task_type=TaskType.REVIEW,
        priority=6,
        requires_any=("backend", "auth", "ai", "database", "frontend", "tests", "integration", "generic_work"),
        depends_on_all_of=("tests", "integration", "docs"),
    ),
    Blueprint(
        key="generic_work",
        title="Implement the requested change",
        description=(
            "Implement exactly what was asked, following the conventions already "
            "established in the repository. Prefer the smallest change that fully works."
        ),
        task_type=TaskType.GENERIC,
        priority=3,
        depends_on=("architecture",),
    ),
)

BY_KEY = {b.key: b for b in BLUEPRINTS}

#: Deliverables that must exist for the plan to make sense at all.
_MINIMUM_DELIVERABLES = ("backend API",)

#: Appended to every real plan, in this order, once implementation work exists.
_TAIL_KEYS = ("tests", "integration", "review")


class TaskDecomposer:
    """
    Builds a validated DAG from a specification.

        dag = await TaskDecomposer(llm=provider).decompose(spec, repo_analysis)
    """

    def __init__(self, llm: Optional[LLMProvider] = None, max_tasks: int = 20):
        self.llm = llm
        self.max_tasks = max_tasks

    # ── public API ──────────────────────────────────────────────────────────

    async def decompose(
        self,
        spec: TaskSpecification,
        repo_analysis: Optional[dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> DAG:
        dag = self._from_blueprints(spec, repo_analysis or {})
        if use_llm and self.llm is not None:
            dag = await self._refine_with_llm(spec, repo_analysis or {}, dag)
        return self._finalise(dag, spec)

    # ── deterministic core ──────────────────────────────────────────────────

    def _from_blueprints(
        self, spec: TaskSpecification, repo: dict[str, Any]
    ) -> DAG:
        kept = self._resolve_active(spec, repo)
        active_keys = {b.key for b in kept}

        tasks: dict[str, Task] = {}
        key_to_id: dict[str, str] = {}
        for blueprint in kept:
            task = blueprint.build(spec)
            task.title = self._personalise(blueprint.title, spec)
            key_to_id[blueprint.key] = task.id
            tasks[task.id] = task

        for blueprint in kept:
            task = tasks[key_to_id[blueprint.key]]
            wanted = set(blueprint.depends_on) | set(blueprint.depends_on_all_of)
            task.dependencies = [
                key_to_id[dep] for dep in wanted if dep in active_keys and dep != blueprint.key
            ]
            task.input_context["acceptance_criteria"] = spec.acceptance_criteria
            task.input_context["constraints"] = spec.constraints
            task.input_context["original_request"] = spec.raw_request
            if spec.llm_plan:
                task.input_context["llm_plan"] = spec.llm_plan

        dag = DAG(tasks=tasks)
        dag.rebuild_edges()
        return dag

    def _resolve_active(
        self, spec: TaskSpecification, repo: dict[str, Any]
    ) -> list[Blueprint]:
        """
        Select blueprints and iterate to a fixed point.

        A single pass is not enough: dropping `tests` can invalidate
        `integration`, which invalidates `review`. Without iterating, a node
        whose parent was pruned silently becomes an orphan root and runs first
        in the graph -- which is exactly the bug this loop exists to prevent.
        """
        wanted = {b.key for b in self._select_blueprints(spec, repo)}
        for _ in range(len(BLUEPRINTS) + 1):
            active_keys = {
                b.key
                for b in BLUEPRINTS
                if b.key in wanted
                and (not b.requires_any or any(r in wanted for r in b.requires_any))
            }
            new_wanted = {
                b.key
                for b in BLUEPRINTS
                if b.key in wanted
                and (
                    not b.requires_any
                    or any(r in active_keys for r in b.requires_any)
                )
            }
            if new_wanted == wanted:
                break
            wanted = new_wanted
        return [b for b in BLUEPRINTS if b.key in wanted]

    def _select_blueprints(
        self, spec: TaskSpecification, repo: dict[str, Any]
    ) -> list[Blueprint]:
        """
        Decide *which* blueprints the request asks for.

        This is a wish list. `_resolve_active` then prunes it to a consistent
        graph, so it is safe for a node to be wished for speculatively here.
        """
        deliverables = set(spec.deliverables)
        types = set(spec.task_types)

        # Any server-side or UI deliverable implies a backend exists to serve it.
        if TaskType.BACKEND not in types and (
            deliverables & {"retrieval pipeline", "user interface", "authentication", "data layer"}
        ):
            deliverables.add("backend API")

        # A request with no recognisable deliverable still needs an
        # implementation node, otherwise there is nothing to execute.
        if not (deliverables & set(_MINIMUM_DELIVERABLES)) and TaskType.BACKEND not in types:
            deliverables.add("generic work")

        # 1. The spine, always.
        wanted: list[str] = ["analysis", "architecture"]

        # 2. Implementation nodes, matched against detected deliverables.
        for blueprint in BLUEPRINTS:
            key = blueprint.key
            if key in {"analysis", "architecture", "generic_work"}:
                continue
            if key in _TAIL_KEYS:
                continue
            if blueprint.deliverable and blueprint.deliverable in deliverables:
                wanted.append(key)
        if "generic work" in deliverables and "generic_work" not in wanted:
            wanted.append("generic_work")

        # 3. The verification tail, added only once there is real work to
        #    verify. Adding it earlier in the iteration order would let it be
        #    dropped by the `len(wanted) >= 3` guard below.
        if len(wanted) >= 3:
            wanted.extend(_TAIL_KEYS)

        return [b for b in BLUEPRINTS if b.key in wanted]

    @staticmethod
    def _personalise(title: str, spec: TaskSpecification) -> str:
        """Prefix the title with the request's own subject for UI clarity."""
        subject = re.sub(r"^\s*(build|create|add|implement|write|develop)\s+", "", spec.title, flags=re.I)
        subject = subject.split(",")[0].strip()
        if not subject or len(subject) > 48:
            return title
        # Plain ASCII separator: these titles are rendered in terminals, logs
        # and JSON payloads, and an em dash mangles under cp1252.
        return f"{title}: {subject}"

    def _finalise(self, dag: DAG, spec: TaskSpecification) -> DAG:
        """Validate, repair and trim the graph. Guarantees a runnable DAG."""
        engine = DAGEngine(dag)
        try:
            engine.validate()
        except CycleDetectedError as exc:
            logger.warning("LLM introduced a cycle; repairing", extra={"cycle": exc.path})
            dag = self._break_cycles(dag, exc.path)
        except Exception as exc:  # missing dependency
            logger.warning("Repairing invalid DAG", extra={"error": str(exc)})
            dag = self._drop_dangling(dag)

        if len(dag.tasks) > self.max_tasks:
            dag = self._trim(dag, self.max_tasks)
        dag.rebuild_edges()
        DAGEngine(dag).validate()
        return dag

    @staticmethod
    def _drop_dangling(dag: DAG) -> DAG:
        for task in dag.tasks.values():
            task.dependencies = [d for d in task.dependencies if d in dag.tasks]
        return dag

    def _break_cycles(self, dag: DAG, path: list[str]) -> DAG:
        """Remove the newest edge in the cycle. Preserves the oldest edges."""
        for dep in reversed(path[:-1]):
            task = dag.tasks.get(dep)
            if task and path[-1] in task.dependencies:
                task.dependencies.remove(path[-1])
                return dag
        return self._drop_dangling(dag)

    def _trim(self, dag: DAG, limit: int) -> DAG:
        """Drop the lowest-priority leaves first, never the spine."""
        while len(dag.tasks) > limit:
            engine = DAGEngine(dag)
            leaves = [tid for tid in engine.leaves() if dag.tasks[tid].priority > 3]
            if not leaves:
                leaves = sorted(
                    engine.leaves(), key=lambda t: -dag.tasks[t].priority
                )
            victim = leaves[0]
            for task in dag.tasks.values():
                task.dependencies = [d for d in task.dependencies if d != victim]
            dag.tasks.pop(victim, None)
        return dag

    # ── optional LLM refinement ─────────────────────────────────────────────

    async def _refine_with_llm(
        self, spec: TaskSpecification, repo: dict[str, Any], dag: DAG
    ) -> DAG:
        """
        Let the model adjust the plan, then re-validate.

        The model may rename, retype and re-prioritise nodes, and add new ones
        with declared dependencies. Anything that breaks the graph is repaired
        by `_finalise`, so this call can never return an invalid DAG.
        """
        from app.llm.opencode import parse_json_response

        current = [
            {
                "id": task.id[:8],
                "key": task.input_context.get("blueprint", ""),
                "title": task.title,
                "type": task.type.value,
                "depends_on": [d[:8] for d in task.dependencies],
            }
            for task in dag.tasks.values()
        ]
        system = (
            "You are a senior engineer refining an implementation plan. You are given "
            "the current task graph. Improve it, but reply with JSON only."
        )
        user = json.dumps(
            {
                "request": spec.raw_request,
                "deliverables": spec.deliverables,
                "current_plan": current,
                "repository": {k: repo.get(k) for k in ("frameworks", "languages", "total_files")},
            },
            indent=2,
        )
        try:
            response = await self.llm.complete(
                user + "\n\nReply with JSON: "
                '{"tasks": [{"id": "<existing id or new>", "title": "...", '
                '"type": "backend|frontend|database|ai_ml|testing|devops|security|'
                'documentation|review|integration|generic", "depends_on": ["<id>"], '
                '"description": "..."}]}',
                system,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Decomposer refinement skipped", extra={"reason": str(exc)[:200]})
            return dag

        parsed = parse_json_response(response.text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tasks"), list):
            return dag
        return self._apply_refinement(dag, parsed["tasks"])

    def _apply_refinement(self, dag: DAG, proposals: list[dict]) -> DAG:
        """Apply model proposals defensively, keyed by the id we showed it.

        A model is never trusted with graph integrity: unknown ids become new
        root nodes, unknown dependency ids are dropped, self-references are
        refused, and `_finalise` re-validates the result before it is returned.
        """
        by_prefix = {task.id[:8]: task for task in dag.tasks.values()}
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            ref = str(proposal.get("id", "")).strip()
            existing = by_prefix.get(ref)
            if existing is not None:
                if isinstance(proposal.get("title"), str) and proposal["title"].strip():
                    existing.title = proposal["title"].strip()[:200]
                if isinstance(proposal.get("description"), str):
                    existing.description = proposal["description"].strip()[:2000]
                if proposal.get("type"):
                    existing.type = Task._coerce_type(proposal["type"])
                if isinstance(proposal.get("depends_on"), list):
                    deps = [
                        by_prefix[d].id
                        for d in proposal["depends_on"]
                        if isinstance(d, str) and d in by_prefix
                    ]
                    # Refuse self-reference; `_finalise` breaks longer cycles.
                    if existing.id not in deps:
                        existing.dependencies = deps
                continue
            # New node proposed by the model.
            if len(dag.tasks) >= self.max_tasks:
                continue
            task = Task(
                id=new_id(),
                title=str(proposal.get("title") or "Additional work")[:200],
                description=str(proposal.get("description") or "")[:2000],
                type=Task._coerce_type(proposal.get("type", "generic")),
                priority=6,
                dependencies=[
                    by_prefix[d].id
                    for d in (proposal.get("depends_on") or [])
                    if isinstance(d, str) and d in by_prefix
                ],
                input_context={"source": "llm_refinement"},
            )
            by_prefix[task.id[:8]] = task
            dag.tasks[task.id] = task
        dag.rebuild_edges()
        return dag

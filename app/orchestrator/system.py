"""
Orchestrator.

Owns the pipeline and nothing else:

    analyse -> plan -> build DAG -> schedule agents in isolated worktrees
            -> integrate -> review -> report

Every decision that must be reproducible -- graph validation, scheduling,
retries, git, tests, integration -- lives in ordinary Python. The LLM is asked
only for understanding and for proposing code, and its output is validated
before it is allowed to change anything.

The orchestrator is intentionally thin. Analysis, decomposition, scheduling,
agent execution and integration are separate collaborators; this class is the
seam between them, not the place their logic lives.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.agents.executor import AgentExecutor
from app.agents.registry import AgentRegistry
from app.agents.selector import AgentSelector
from app.brain.analyzer import TaskAnalyzer
from app.brain.context import ContextManager, RepositorySnapshot, scan_repository
from app.brain.decomposer import TaskDecomposer
from app.brain.repo_analyzer import RepositoryAnalyzer
from app.config import Settings, get_settings
from app.engine.dag import DAGEngine
from app.engine.scheduler import RetryPolicy, Scheduler, SchedulerStats
from app.events import EventBus, EventType, get_event_bus
from app.git.manager import GitManager
from app.integrator.system import Integrator
from app.llm.base import LLMProvider
from app.logging_config import get_logger, orchestration_scope, reset_orchestration_scope
from app.models.domain import (
    DAG,
    ExecutionState,
    FinalStatus,
    IntegrationReport,
    Phase,
    Project,
    TaskStatus,
)
from app.models.orchestration import Orchestration

logger = get_logger("app.orchestrator")


@dataclass
class OrchestrationDeps:
    """Every collaborator the orchestrator needs, in one injectable bundle."""

    registry: AgentRegistry
    git: GitManager
    llm: Optional[LLMProvider] = None
    settings: Optional[Settings] = None
    event_bus: Optional[EventBus] = None
    workspace_root: Optional[str] = None
    repo_analyzer: Optional[RepositoryAnalyzer] = None
    task_analyzer: Optional[TaskAnalyzer] = None
    decomposer: Optional[TaskDecomposer] = None
    selector: Optional[AgentSelector] = None
    context_manager: Optional[ContextManager] = None
    integrator: Optional[Integrator] = None

    def resolved_settings(self) -> Settings:
        return self.settings or get_settings()


class Orchestrator:
    """
    Runs one request end to end.

        orchestrator = Orchestrator(OrchestrationDeps(registry=..., git=...))
        orchestration = await orchestrator.plan(project, request)
        state = await orchestrator.execute(orchestration)
    """

    def __init__(self, deps: OrchestrationDeps):
        self.deps = deps
        self.settings = deps.resolved_settings()
        self.event_bus = deps.event_bus or get_event_bus()
        self.repo_analyzer = deps.repo_analyzer or RepositoryAnalyzer(llm=deps.llm)
        self.task_analyzer = deps.task_analyzer or TaskAnalyzer(llm=deps.llm)
        self.decomposer = deps.decomposer or TaskDecomposer(llm=deps.llm)
        self.selector = deps.selector or AgentSelector()
        self.snapshot: RepositorySnapshot = RepositorySnapshot()
        self._live_scheduler: Optional[Scheduler] = None
        self._manager = None

    # ── convenience constructors ────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        registry: AgentRegistry,
        git: GitManager,
        llm: Optional[LLMProvider] = None,
        settings: Optional[Settings] = None,
        event_bus: Optional[EventBus] = None,
        **kwargs: Any,
    ) -> "Orchestrator":
        return cls(
            OrchestrationDeps(
                registry=registry,
                git=git,
                llm=llm,
                settings=settings,
                event_bus=event_bus,
                **kwargs,
            )
        )

    # ── phase 1: analysis & planning ────────────────────────────────────────

    async def plan(self, project: Project, user_request: str) -> Orchestration:
        """
        Analyse, plan and build the DAG. No code runs yet.

        Split out from `execute` so the UI can show the plan and let a human
        inspect it before a single agent is dispatched.
        """
        token = orchestration_scope("")
        try:
            await self._phase(Phase.ANALYSIS)
            self.snapshot = scan_repository(project.local_path)
            repo_analysis = await self.repo_analyzer.analyze(project.local_path)
            self.snapshot = repo_analysis.snapshot or self.snapshot

            await self._phase(Phase.PLANNING)
            spec = await self.task_analyzer.analyze(
                user_request, repo_analysis.to_dict(), use_llm=self.deps.llm is not None
            )
            dag = await self.decomposer.decompose(
                spec, repo_analysis.to_dict(), use_llm=self.deps.llm is not None
            )

            state = ExecutionState(
                project_id=project.id,
                original_task=user_request,
                repository=project.local_path,
                dag=dag,
                tasks=dict(dag.tasks),
                repository_analysis=repo_analysis.to_dict(),
                task_analysis=spec.to_dict(),
            )
            state.current_phase = Phase.EXECUTION
            engine = DAGEngine(dag)
            engine.validate()

            orchestration = Orchestration(
                state=state,
                name=spec.title[:120] or "Orchestration",
                total_tasks=len(dag.tasks),
                base_branch=await self.deps.git.default_branch(),
                workspace_root=self.deps.workspace_root or self.settings.workspace_root,
            )
            await self.event_bus.publish(
                orchestration.id,
                EventType.DAG_CREATED,
                tasks=[
                    {
                        "id": task.id,
                        "title": task.title,
                        "type": task.type.value,
                        "priority": task.priority,
                        "dependencies": task.dependencies,
                        "agent": self.selector.select(task).role,
                    }
                    for task in dag.tasks.values()
                ],
                waves=engine.parallel_waves(),
                critical_path=engine.critical_path(),
                max_parallelism=engine.estimated_parallelism(
                    self.settings.max_parallel_tasks
                ),
                summary=engine.summary(),
                repository=repo_analysis.summary,
            )
            return orchestration
        finally:
            with contextlib.suppress(Exception):
                reset_orchestration_scope(token)

    # ── phase 2: execution ──────────────────────────────────────────────────

    def runner_factory(self, orchestration: Orchestration):
        """
        Build the per-task coroutine the Scheduler will call.

        One executor is shared across all tasks (it is stateless per call), and
        each task gets its own worktree inside it. The closure wires the
        scheduler's `Task -> await None` contract to the executor's richer API.
        """
        executor = self._build_executor(orchestration)

        async def run_task(task) -> None:
            attempt = task.retry_count + 1
            await executor.run_task(task, attempt=attempt)

        return run_task

    def bind(self, orchestration: Orchestration, manager=None):
        """
        The run callable `ExecutionManager` drives.

        Returns a coroutine function that runs the *whole* pipeline --
        schedule every task, then integrate and review -- rather than just the
        scheduling half. `ExecutionManager` owns lifetime, not logic, so the
        integration step must not live on the other side of this boundary.

        `manager` is optional; when supplied, the live scheduler is registered
        with it so pause / resume / cancel can reach the running graph.
        """

        async def run(on_state_change=None):
            return await self.execute(
                orchestration,
                on_state_change=on_state_change,
                manager=manager,
            )

        return run

    def _build_executor(self, orchestration: Orchestration) -> AgentExecutor:
        context_manager = self.deps.context_manager or ContextManager(self.snapshot)
        return AgentExecutor(
            registry=self.deps.registry,
            selector=self.selector,
            state=orchestration.state,
            context_manager=context_manager,
            git_manager=self.deps.git,
            workspace_root=orchestration.workspace_root or self.settings.workspace_root,
            event_bus=self.event_bus,
            orchestration_id=orchestration.id,
            base_branch=orchestration.base_branch,
        )

    async def execute(
        self,
        orchestration: Orchestration,
        on_state_change=None,
        manager=None,
    ) -> ExecutionState:
        """
        Schedule and run every task, then integrate the results.

        The DAG engine owns ordering and retries; this method owns the phase
        transitions and the final integration step.
        """
        token = orchestration_scope(orchestration.id)
        state = orchestration.state
        scheduler: Scheduler | None = None
        try:
            state.current_phase = Phase.EXECUTION
            await self.event_bus.publish(
                orchestration.id, EventType.ORCHESTRATION_PHASE, phase=Phase.EXECUTION.value
            )
            engine = DAGEngine(state.dag)
            runner = self.runner_factory(orchestration)
            scheduler = Scheduler(
                engine,
                runner,
                orchestration_id=orchestration.id,
                max_concurrency=self.settings.max_parallel_tasks,
                retry_policy=RetryPolicy(
                    max_retries=self.settings.max_task_retries,
                    base_delay=self.settings.retry_backoff_seconds,
                    multiplier=self.settings.retry_backoff_multiplier,
                ),
                event_bus=self.event_bus,
                task_timeout=self.settings.task_timeout_seconds,
                on_state_change=on_state_change,
            )
            self._live_scheduler = scheduler
            if manager is not None:
                manager.register_scheduler(orchestration.id, scheduler)
            stats = await scheduler.run()
            await self._after_execution(orchestration, stats)
            return state
        finally:
            self._live_scheduler = None
            with contextlib.suppress(Exception):
                reset_orchestration_scope(token)

    @property
    def live_scheduler(self) -> Optional[Scheduler]:
        """The scheduler of the run currently in flight, if any."""
        return self._live_scheduler

    async def _after_execution(
        self, orchestration: Orchestration, stats: SchedulerStats
    ) -> None:
        state = orchestration.state
        if stats.error:
            state.record_error(f"Scheduling failed: {stats.error}")

        succeeded = [t for t in state.tasks.values() if t.status == TaskStatus.SUCCESS]
        if not succeeded:
            state.final_status = FinalStatus.FAILED
            state.current_phase = Phase.FAILED
            if not state.errors:
                state.errors.append("No task completed successfully")
            return

        state.current_phase = Phase.INTEGRATION
        await self.event_bus.publish(
            orchestration.id,
            EventType.ORCHESTRATION_PHASE,
            phase=Phase.INTEGRATION.value,
        )
        report = await self._integrate(orchestration)

        state.current_phase = Phase.REVIEW if report else Phase.COMPLETED
        await self.event_bus.publish(
            orchestration.id,
            EventType.ORCHESTRATION_PHASE,
            phase=state.current_phase.value,
        )
        if report is not None:
            state.integration_report = report
            if not report.success:
                for finding in report.regressions:
                    state.record_error(finding)
        state.touch()

    async def _integrate(self, orchestration: Orchestration) -> Optional[IntegrationReport]:
        state = orchestration.state
        if not self._should_integrate(state):
            logger.info("Skipping integration: no branches to merge")
            return None
        integrator = self.deps.integrator or Integrator(
            git_manager=self.deps.git,
            settings=self.settings,
            event_bus=self.event_bus,
            orchestration_id=orchestration.id,
            llm=self.deps.llm,
        )
        try:
            return await integrator.integrate(state)
        except Exception as exc:  # noqa: BLE001 - a failed merge is a result
            logger.exception("Integration failed")
            state.record_error(f"Integration failed: {exc}")
            state.final_status = FinalStatus.FAILED
            return None

    @staticmethod
    def _should_integrate(state: ExecutionState) -> bool:
        """Nothing to merge unless at least one agent actually committed."""
        return any(
            result.status == TaskStatus.SUCCESS and result.branch
            for result in state.agent_results.values()
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    async def _phase(self, phase: Phase) -> None:
        # Phase changes are published by ExecutionManager where an id exists;
        # here we only advance the model so planning-only callers see progress.
        return None

    async def emit_phase(self, orchestration_id: str, phase: Phase) -> None:
        await self.event_bus.publish(
            orchestration_id, EventType.ORCHESTRATION_PHASE, phase=phase.value
        )

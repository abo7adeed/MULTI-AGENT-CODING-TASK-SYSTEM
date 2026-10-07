"""
Orchestrator tests.

The orchestrator is the seam between the collaborators, so these tests mostly
check the *seam*: that planning produces a valid, inspectable plan before
anything runs, that execution drives the scheduler, that integration is gated
correctly, and that a failure anywhere ends the run honestly rather than
reporting a green result.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.agents.mock import build_mock_roster
from app.agents.registry import AgentRegistry
from app.brain.analyzer import TaskAnalyzer
from app.brain.decomposer import TaskDecomposer
from app.config import Settings
from app.engine.dag import DAGEngine
from app.engine.execution import ExecutionManager
from app.events import EventBus, EventType
from app.git.manager import GitManager, is_git_installed
from app.llm.mock import MockLLMProvider
from app.models.domain import (
    FinalStatus,
    OrchestrationStatus,
    Phase,
    Project,
    Task,
    TaskStatus,
    TaskType,
)
from app.orchestrator.system import OrchestrationDeps, Orchestrator

pytestmark = pytest.mark.skipif(
    not is_git_installed(), reason="git is not installed on PATH"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("def main():\n    return 'base'\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def repo_git(repo: Path) -> GitManager:
    manager = GitManager(repo)
    asyncio.run(manager.init_repo())
    asyncio.run(manager.commit_all(repo, "base"))
    return manager


@pytest.fixture
def project(repo: Path) -> Project:
    return Project(name="demo", local_path=str(repo), base_branch="main")


@pytest.fixture
def mock_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register_all(build_mock_roster(delay=0.0))
    return registry


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        workspace_root=str(tmp_path / "workspaces"),
        state_db_path=str(tmp_path / "state.db"),
        max_parallel_tasks=4,
        max_task_retries=0,
        retry_backoff_seconds=0.01,
        retry_backoff_multiplier=1.0,
        task_timeout_seconds=60,
        sandbox_enabled=False,
        test_timeout_seconds=120,
    )


def make_orchestrator(
    project_git: GitManager,
    registry: AgentRegistry,
    settings: Settings,
    bus: EventBus | None = None,
    llm=None,
    **kwargs,
) -> Orchestrator:
    return Orchestrator.build(
        registry=registry,
        git=project_git,
        settings=settings,
        event_bus=bus or EventBus(),
        llm=llm,
        **kwargs,
    )


# ── planning ────────────────────────────────────────────────────────────────


class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_produces_a_valid_dag_without_running_anything(
        self, project, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        assert orchestration.total_tasks == len(orchestration.state.dag.tasks)
        assert orchestration.total_tasks > 0
        DAGEngine(orchestration.state.dag).validate()
        # Nothing has executed yet.
        assert orchestration.state.agent_results == {}
        assert orchestration.state.final_status is FinalStatus.PENDING

    @pytest.mark.asyncio
    async def test_the_plan_records_the_analysis(
        self, project, repo_git, mock_registry, settings
    ):
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings
        ).plan(project, "Add a health endpoint")
        state = orchestration.state
        assert state.repository_analysis["total_files"] > 0
        assert state.task_analysis["raw_request"] == "Add a health endpoint"
        assert state.task_analysis["task_types"]

    @pytest.mark.asyncio
    async def test_the_name_comes_from_the_request(
        self, project, repo_git, mock_registry, settings
    ):
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings
        ).plan(project, "Add a health endpoint")
        assert "health" in orchestration.name.lower()

    @pytest.mark.asyncio
    async def test_the_base_branch_is_detected_from_the_repository(
        self, project, repo_git, mock_registry, settings
    ):
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings
        ).plan(project, "Add a health endpoint")
        assert orchestration.base_branch == "main"

    @pytest.mark.asyncio
    async def test_the_workspace_root_comes_from_settings(
        self, project, repo_git, mock_registry, settings
    ):
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings
        ).plan(project, "Add a health endpoint")
        assert orchestration.workspace_root == settings.workspace_root

    @pytest.mark.asyncio
    async def test_a_dag_created_event_carries_everything_the_ui_needs(
        self, project, repo_git, mock_registry, settings
    ):
        bus = EventBus()
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings, bus=bus
        ).plan(project, "Add a health endpoint")
        events = [e for e in bus.recent(orchestration.id) if e.type is EventType.DAG_CREATED]
        assert len(events) == 1
        data = events[0].data
        assert len(data["tasks"]) == orchestration.total_tasks
        assert set(data["tasks"][0]) >= {"id", "title", "type", "priority", "agent"}
        assert data["waves"]
        assert data["summary"]
        assert data["repository"]

    @pytest.mark.asyncio
    async def test_every_task_gets_an_agent_assigned(
        self, project, repo_git, mock_registry, settings
    ):
        bus = EventBus()
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings, bus=bus
        ).plan(project, "Build a full stack app with auth and tests")
        event = [e for e in bus.recent(orchestration.id) if e.type is EventType.DAG_CREATED][0]
        agents = {t["agent"] for t in event.data["tasks"]}
        assert agents
        for agent in agents:
            assert mock_registry.get(agent) is not None, f"no agent registered for {agent}"

    @pytest.mark.asyncio
    async def test_an_empty_repository_still_plans(self, tmp_path, mock_registry, settings):
        empty = tmp_path / "greenfield"
        empty.mkdir()
        git = GitManager(empty)
        await git.init_repo()
        await git.commit_all(empty, "empty")
        project = Project(name="new", local_path=str(empty), base_branch="main")
        orchestration = await make_orchestrator(
            git, mock_registry, settings
        ).plan(project, "Build a new todo API with tests")
        assert orchestration.total_tasks > 0
        DAGEngine(orchestration.state.dag).validate()

    @pytest.mark.asyncio
    async def test_a_missing_repository_path_does_not_crash(
        self, mock_registry, settings
    ):
        project = Project(name="gone", local_path="does/not/exist", base_branch="main")
        git = GitManager("does/not/exist")
        orchestration = await make_orchestrator(
            git, mock_registry, settings
        ).plan(project, "Add a health endpoint")
        assert orchestration.total_tasks > 0

    @pytest.mark.asyncio
    async def test_planning_is_repeatable_for_the_same_request(
        self, project, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        first = await orchestrator.plan(project, "Add a health endpoint")
        second = await orchestrator.plan(project, "Add a health endpoint")
        assert first.total_tasks == second.total_tasks
        assert {t.title for t in first.state.dag.tasks.values()} == {
            t.title for t in second.state.dag.tasks.values()
        }

    @pytest.mark.asyncio
    async def test_an_llm_is_only_consulted_when_configured(
        self, project, repo_git, mock_registry, settings
    ):
        with_llm = MockLLMProvider()
        await make_orchestrator(repo_git, mock_registry, settings, llm=with_llm).plan(
            project, "Add a health endpoint"
        )
        assert with_llm.prompts

    @pytest.mark.asyncio
    async def test_planning_works_with_no_llm_at_all(
        self, project, repo_git, mock_registry, settings
    ):
        orchestration = await make_orchestrator(
            repo_git, mock_registry, settings, llm=None
        ).plan(project, "Add a health endpoint")
        assert orchestration.state.task_analysis["source"] == "heuristic"
        DAGEngine(orchestration.state.dag).validate()


# ── execution ───────────────────────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_a_full_run_reaches_a_terminal_state(
        self, project, repo, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.execute(orchestration)
        assert state.final_status is not FinalStatus.PENDING
        assert state.tasks, "the run should have produced tasks"
        assert any(t.status == TaskStatus.SUCCESS for t in state.tasks.values())

    @pytest.mark.asyncio
    async def test_phase_transitions_are_published(
        self, project, repo_git, mock_registry, settings
    ):
        bus = EventBus()
        orchestrator = make_orchestrator(repo_git, mock_registry, settings, bus=bus)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        await orchestrator.execute(orchestration)
        phases = [
            e.data["phase"]
            for e in bus.recent(orchestration.id)
            if e.type is EventType.ORCHESTRATION_PHASE
        ]
        assert Phase.EXECUTION.value in phases
        assert Phase.INTEGRATION.value in phases

    @pytest.mark.asyncio
    async def test_the_live_scheduler_is_cleared_afterwards(
        self, project, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        assert orchestrator.live_scheduler is None
        await orchestrator.execute(orchestration)
        assert orchestrator.live_scheduler is None

    @pytest.mark.asyncio
    async def test_on_state_change_is_called(
        self, project, repo_git, mock_registry, settings
    ):
        calls: list[int] = []
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")

        async def on_change():
            calls.append(1)

        await orchestrator.execute(orchestration, on_state_change=on_change)
        assert calls, "the persistence hook should have been invoked"

    @pytest.mark.asyncio
    async def test_a_manager_receives_the_scheduler_for_control(
        self, project, repo_git, mock_registry, settings
    ):
        from app.engine.scheduler import Scheduler
        from app.models.store import InMemoryStateStore

        store = InMemoryStateStore()
        bus = EventBus()
        manager = ExecutionManager(store=store, event_bus=bus, settings=settings)
        orchestrator = make_orchestrator(repo_git, mock_registry, settings, bus=bus)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")

        seen: list[str] = []

        async def on_change():
            seen.append(orchestration.id)
            if not seen[1:]:
                # Reach into the manager mid-run: the scheduler must be reachable.
                assert manager.is_running(orchestration.id) is True

        await orchestrator.execute(orchestration, on_state_change=on_change, manager=manager)
        assert seen

    @pytest.mark.asyncio
    async def test_every_successful_task_left_a_branch_and_a_commit(
        self, project, repo, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.execute(orchestration)
        successful = [
            r for r in state.agent_results.values() if r.status is TaskStatus.SUCCESS
        ]
        assert successful
        for result in successful:
            assert result.branch, f"{result.task_id} succeeded without a branch"
            assert result.commit

    @pytest.mark.asyncio
    async def test_state_changes_are_touched(
        self, project, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        before = orchestration.state.updated_at
        await asyncio.sleep(0.01)
        await orchestrator.execute(orchestration)
        assert orchestration.state.updated_at > before


# ── runner and binding ──────────────────────────────────────────────────────


class TestRunnerAndBind:
    @pytest.mark.asyncio
    async def test_runner_factory_drives_the_executor(
        self, project, repo, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        run_task = orchestrator.runner_factory(orchestration)
        task = next(t for t in orchestration.state.dag.tasks.values() if t.type is TaskType.BACKEND)
        await run_task(task)
        assert task.id in orchestration.state.agent_results

    @pytest.mark.asyncio
    async def test_the_attempt_number_follows_the_retry_count(
        self, project, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        run_task = orchestrator.runner_factory(orchestration)
        task = next(iter(orchestration.state.dag.tasks.values()))
        task.retry_count = 2
        await run_task(task)
        assert orchestration.state.agent_results[task.id].attempt == 3

    @pytest.mark.asyncio
    async def test_bind_runs_the_whole_pipeline_including_integration(
        self, project, repo, repo_git, mock_registry, settings
    ):
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.bind(orchestration)()
        assert state.final_status is not FinalStatus.PENDING
        assert state.integration_report is not None

    @pytest.mark.asyncio
    async def test_bind_forwards_the_state_change_callback(
        self, project, repo_git, mock_registry, settings
    ):
        seen: list[int] = []
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")

        async def on_change():
            seen.append(1)

        await orchestrator.bind(orchestration)(on_state_change=on_change)
        assert seen


# ── integration gating ──────────────────────────────────────────────────────


class TestIntegrationGating:
    def test_nothing_to_merge_means_no_integration(self):
        from app.models.domain import AgentResult, ExecutionState

        state = ExecutionState(
            project_id="p", original_task="o", repository="r"
        )
        assert Orchestrator._should_integrate(state) is False

    def test_a_successful_result_with_a_branch_triggers_integration(self):
        from app.models.domain import AgentResult, ExecutionState, new_id

        state = ExecutionState(project_id="p", original_task="o", repository="r")
        tid = new_id()
        state.agent_results[tid] = AgentResult(
            task_id=tid, status=TaskStatus.SUCCESS, summary="ok", branch="agent/a"
        )
        assert Orchestrator._should_integrate(state) is True

    def test_a_success_without_a_branch_does_not_trigger_integration(self):
        from app.models.domain import AgentResult, ExecutionState, new_id

        state = ExecutionState(project_id="p", original_task="o", repository="r")
        tid = new_id()
        state.agent_results[tid] = AgentResult(
            task_id=tid, status=TaskStatus.SUCCESS, summary="ok"
        )
        assert Orchestrator._should_integrate(state) is False

    def test_a_failed_result_with_a_branch_does_not_trigger_integration(self):
        from app.models.domain import AgentResult, ExecutionState, new_id

        state = ExecutionState(project_id="p", original_task="o", repository="r")
        tid = new_id()
        state.agent_results[tid] = AgentResult(
            task_id=tid, status=TaskStatus.FAILED, summary="no", branch="agent/a"
        )
        assert Orchestrator._should_integrate(state) is False

    @pytest.mark.asyncio
    async def test_an_injected_integrator_is_used(
        self, project, repo, repo_git, mock_registry, settings
    ):
        from app.integrator.system import Integrator
        from app.integrator.test_runner import TestRunner

        class Spy(Integrator):
            calls = 0

            async def integrate(self, state):
                Spy.calls += 1
                return await super().integrate(state)

        orchestrator = make_orchestrator(
            repo_git,
            mock_registry,
            settings,
            integrator=Spy(
                git_manager=repo_git,
                test_runner=TestRunner(timeout=120),
                settings=settings,
            ),
        )
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        await orchestrator.execute(orchestration)
        assert Spy.calls == 1

    @pytest.mark.asyncio
    async def test_an_integrator_that_raises_is_reported_not_propagated(
        self, project, repo_git, mock_registry, settings
    ):
        class Broken:
            async def integrate(self, state):
                raise RuntimeError("merge exploded")

        orchestrator = make_orchestrator(
            repo_git, mock_registry, settings, integrator=Broken()
        )
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.execute(orchestration)
        assert state.final_status is FinalStatus.FAILED
        assert any("Integration failed" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_integration_is_skipped_when_no_agent_committed(
        self, project, repo_git, mock_registry, settings
    ):
        from app.models.domain import AgentResult, new_id

        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        # Strip the branches so nothing is mergeable.
        for task_id in list(orchestration.state.agent_results):
            pass
        state = orchestration.state
        state.agent_results = {
            tid: AgentResult(
                task_id=tid,
                status=TaskStatus.SUCCESS,
                summary="analysis only",
                files_changed=[],
            )
            for tid in list(state.tasks)[:1]
        }
        await orchestrator._integrate(orchestration)
        assert state.integration_report is None


# ── failure handling ────────────────────────────────────────────────────────


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_a_run_where_nothing_succeeds_fails_honestly(
        self, project, repo, repo_git, settings
    ):
        """Every agent fails: the run must not claim success."""
        from app.models.domain import AgentResult

        registry = AgentRegistry()
        registry.register_all(
            build_mock_roster(delay=0.0, succeed=False)
        )

        orchestrator = make_orchestrator(repo_git, registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.execute(orchestration)
        assert state.final_status is FinalStatus.FAILED
        assert state.current_phase is Phase.FAILED
        assert "No task completed successfully" in state.errors

    @pytest.mark.asyncio
    async def test_a_scheduler_error_is_recorded(
        self, project, repo_git, mock_registry, settings, monkeypatch
    ):
        from app.engine import scheduler as scheduler_module

        original = scheduler_module.Scheduler.run

        async def boom(self):
            return scheduler_module.SchedulerStats(error="the scheduler gave up")

        monkeypatch.setattr(scheduler_module.Scheduler, "run", boom)
        orchestrator = make_orchestrator(repo_git, mock_registry, settings)
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.execute(orchestration)
        assert any("the scheduler gave up" in e for e in state.errors)
        assert original is not None

    @pytest.mark.asyncio
    async def test_report_regressions_become_state_errors(
        self, project, repo, repo_git, mock_registry, settings
    ):
        from app.integrator.system import Integrator

        class FailsReview(Integrator):
            async def integrate(self, state):
                report = await super().integrate(state)
                report.regressions.append("Critical review finding: hardcoded secret")
                report.success = False
                return report

        orchestrator = make_orchestrator(
            repo_git,
            mock_registry,
            settings,
            integrator=FailsReview(
                git_manager=repo_git, test_runner=None, settings=settings
            ),
        )
        orchestration = await orchestrator.plan(project, "Add a health endpoint")
        state = await orchestrator.execute(orchestration)
        assert any("hardcoded secret" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_a_verdict_the_pipeline_reached_is_not_overwritten_by_task_counts(
        self, project, repo_git, mock_registry, settings
    ):
        """
        Every task can report SUCCESS while the integrator merges nothing.
        The manager's task tally must not turn that into a clean success.
        """
        from app.models.domain import AgentResult, ExecutionState
        from app.models.orchestration import Orchestration
        from app.models.store import InMemoryStateStore

        store = InMemoryStateStore()
        bus = EventBus()
        manager = ExecutionManager(store=store, event_bus=bus, settings=settings)

        state = ExecutionState(
            project_id="p", original_task="o", repository=str(project.local_path)
        )
        task = Task(title="t", type=TaskType.BACKEND, branch="agent/a")
        state.tasks[task.id] = task
        task.status = TaskStatus.SUCCESS
        state.agent_results[task.id] = AgentResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            summary="ok",
            branch="agent/a",
            files_changed=[],
        )
        state.final_status = FinalStatus.FAILED  # as the integrator would set it
        orchestration = Orchestration(state=state, name="run", total_tasks=1)

        await manager._finalise(orchestration, None, None)
        assert state.final_status is FinalStatus.FAILED
        assert orchestration.status is OrchestrationStatus.FAILED

    @pytest.mark.asyncio
    async def test_a_genuinely_successful_run_is_still_reported_as_success(
        self, project, repo_git, mock_registry, settings
    ):
        from app.models.domain import AgentResult, ExecutionState
        from app.models.orchestration import Orchestration
        from app.models.store import InMemoryStateStore

        store = InMemoryStateStore()
        bus = EventBus()
        manager = ExecutionManager(store=store, event_bus=bus, settings=settings)
        state = ExecutionState(
            project_id="p", original_task="o", repository=str(project.local_path)
        )
        task = Task(title="t", type=TaskType.BACKEND, branch="agent/a")
        state.tasks[task.id] = task
        task.status = TaskStatus.SUCCESS
        state.agent_results[task.id] = AgentResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            summary="ok",
            branch="agent/a",
            files_changed=["a.py"],
        )
        orchestration = Orchestration(state=state, name="run", total_tasks=1)
        await manager._finalise(orchestration, None, None)
        assert state.final_status is FinalStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_a_partly_successful_run_is_partial_not_failed(
        self, project, repo_git, mock_registry, settings
    ):
        from app.models.domain import AgentResult, ExecutionState
        from app.models.orchestration import Orchestration
        from app.models.store import InMemoryStateStore

        store = InMemoryStateStore()
        manager = ExecutionManager(
            store=store, event_bus=EventBus(), settings=settings
        )
        state = ExecutionState(
            project_id="p", original_task="o", repository=str(project.local_path)
        )
        good = Task(title="good", type=TaskType.BACKEND, branch="agent/a")
        bad = Task(title="bad", type=TaskType.BACKEND, branch="agent/b")
        state.tasks[good.id] = good
        state.tasks[bad.id] = bad
        good.status = TaskStatus.SUCCESS
        bad.status = TaskStatus.FAILED
        state.agent_results[good.id] = AgentResult(
            task_id=good.id, status=TaskStatus.SUCCESS, summary="ok", branch="agent/a"
        )
        orchestration = Orchestration(state=state, name="run", total_tasks=2)
        await manager._finalise(orchestration, None, None)
        assert state.final_status is FinalStatus.PARTIAL_SUCCESS

    @pytest.mark.asyncio
    async def test_emit_phase_publishes(self, project, repo_git, mock_registry, settings):
        bus = EventBus()
        orchestrator = make_orchestrator(repo_git, mock_registry, settings, bus=bus)
        await orchestrator.emit_phase("orch-1", Phase.PLANNING)
        events = bus.recent("orch-1")
        assert events[-1].type is EventType.ORCHESTRATION_PHASE
        assert events[-1].data["phase"] == Phase.PLANNING.value


# ── dependency wiring ───────────────────────────────────────────────────────


class TestDeps:
    def test_build_constructs_the_deps_bundle(self, repo_git, mock_registry, settings):
        orchestrator = Orchestrator.build(
            registry=mock_registry, git=repo_git, settings=settings
        )
        assert isinstance(orchestrator.deps, OrchestrationDeps)
        assert orchestrator.deps.git is repo_git
        assert orchestrator.deps.registry is mock_registry

    def test_defaults_are_built_when_not_injected(self, repo_git, mock_registry, settings):
        orchestrator = Orchestrator.build(
            registry=mock_registry, git=repo_git, settings=settings
        )
        assert orchestrator.repo_analyzer is not None
        assert orchestrator.task_analyzer is not None
        assert orchestrator.decomposer is not None
        assert orchestrator.selector is not None

    def test_injected_collaborators_are_used_verbatim(self, repo_git, mock_registry, settings):
        analyzer = TaskAnalyzer()
        decomposer = TaskDecomposer()
        orchestrator = Orchestrator.build(
            registry=mock_registry,
            git=repo_git,
            settings=settings,
            task_analyzer=analyzer,
            decomposer=decomposer,
        )
        assert orchestrator.task_analyzer is analyzer
        assert orchestrator.decomposer is decomposer

    def test_resolved_settings_falls_back_to_the_global(self, repo_git, mock_registry):
        deps = OrchestrationDeps(registry=mock_registry, git=repo_git)
        assert deps.resolved_settings() is not None


# ── one run per repository ──────────────────────────────────────────────────


class TestRepositoryExclusivity:
    """
    Isolation is per repository, not per run: every agent commits into one
    shared `.git`, and the integrator merges, resets and checks out the target
    branch. Two live runs against the same path therefore interleave inside a
    single repository and corrupt each other's merges, so the second one is
    refused rather than started.
    """

    @staticmethod
    def _orchestration(repo_path: str, name: str):
        from app.models.domain import ExecutionState
        from app.models.orchestration import Orchestration

        state = ExecutionState(
            project_id="p", original_task=name, repository=repo_path
        )
        return Orchestration(state=state, name=name)

    @pytest.mark.asyncio
    async def test_a_second_run_on_the_same_repository_is_refused(
        self, repo_git, settings
    ):
        from app.engine.execution import RepositoryBusy
        from app.models.store import InMemoryStateStore

        manager = ExecutionManager(
            store=InMemoryStateStore(), event_bus=EventBus(), settings=settings
        )
        gate = asyncio.Event()
        repo = repo_git.repo_path

        async def run_fn(on_state_change=None):
            await gate.wait()

        first = await manager.start(self._orchestration(repo, "first"), run_fn)
        with pytest.raises(RepositoryBusy) as excinfo:
            await manager.start(self._orchestration(repo, "second"), run_fn)
        assert "already running" in str(excinfo.value)
        assert repo in str(excinfo.value)

        # A different repository is unaffected.
        elsewhere = await manager.start(
            self._orchestration(repo + "-elsewhere", "elsewhere"), run_fn
        )
        assert manager.is_running(elsewhere.id)

        gate.set()
        assert await manager.wait(first.id, timeout=10) is True
        assert await manager.wait(elsewhere.id, timeout=10) is True

        # The repository is released with the run, not held for the life of the
        # process: a repository must not become permanently unusable.
        again = await manager.start(self._orchestration(repo, "third"), run_fn)
        assert await manager.wait(again.id, timeout=10) is True

    @pytest.mark.asyncio
    async def test_the_same_path_spelled_differently_is_the_same_repository(
        self, repo_git, settings
    ):
        from app.engine.execution import RepositoryBusy
        from app.models.store import InMemoryStateStore

        manager = ExecutionManager(
            store=InMemoryStateStore(), event_bus=EventBus(), settings=settings
        )
        gate = asyncio.Event()
        repo = repo_git.repo_path

        async def run_fn(on_state_change=None):
            await gate.wait()

        first = await manager.start(self._orchestration(repo, "first"), run_fn)
        # A trailing separator and a `.` segment name the same directory.
        with pytest.raises(RepositoryBusy):
            await manager.start(
                self._orchestration(repo + "/./", "second"), run_fn
            )
        gate.set()
        assert await manager.wait(first.id, timeout=10) is True

    @pytest.mark.asyncio
    async def test_a_null_repository_is_not_treated_as_a_shared_one(
        self, settings
    ):
        """Runs with no repository must not all collide on the empty string."""
        from app.models.store import InMemoryStateStore

        manager = ExecutionManager(
            store=InMemoryStateStore(), event_bus=EventBus(), settings=settings
        )
        gate = asyncio.Event()

        async def run_fn(on_state_change=None):
            await gate.wait()

        one = await manager.start(self._orchestration("", "one"), run_fn)
        two = await manager.start(self._orchestration("", "two"), run_fn)
        assert manager.is_running(one.id) and manager.is_running(two.id)
        gate.set()
        assert await manager.wait(one.id, timeout=10) is True
        assert await manager.wait(two.id, timeout=10) is True

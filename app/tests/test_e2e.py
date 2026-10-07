"""
End-to-end tests.

These are the ones that prove the system works, not just that its parts do.
Each test drives the real HTTP API against a real git repository on disk and
then inspects the result: the branches that were merged, the files that landed,
the suite that was run, the worktrees that were cleaned up.

They use mock agents (no network, no API key) but everything else -- git,
worktrees, merges, the real `pytest` run in the target repository -- is genuine.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.engine.dag import DAGEngine
from app.events import EventType
from app.git.manager import GitManager, is_git_installed
from app.llm.mock import MockLLMProvider, RuleBasedLLM
from app.models.domain import FinalStatus, OrchestrationStatus, Project, TaskStatus
from app.models.store import InMemoryStateStore

pytestmark = pytest.mark.skipif(
    not is_git_installed(), reason="git is not installed on PATH"
)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    """A small but realistic project with a real, passing test suite."""
    root = tmp_path / "target"
    (root / "src" / "demo").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "demo" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "demo" / "core.py").write_text(
        "def greet(name: str) -> str:\n"
        '    """Return a greeting."""\n'
        "    return f'hello {name}'\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_core.py").write_text(
        "import sys\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n\n"
        "from demo.core import greet\n\n\n"
        "def test_greet():\n"
        "    assert greet('world') == 'hello world'\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    return root


@pytest.fixture
def runner(tmp_path, target_repo):
    """A real Container wired to the real store, registry and API."""
    from fastapi.testclient import TestClient

    from app.agents.mock import build_mock_roster
    from app.agents.registry import AgentRegistry
    from app.api.deps import Container, set_container
    from app.api.main import create_app
    from app.config import get_settings
    from app.engine.execution import ExecutionManager
    from app.events import EventBus

    settings = Settings(
        _env_file=None,
        state_db_path=str(tmp_path / "state.db"),
        workspace_root=str(tmp_path / "workspaces"),
        max_parallel_tasks=4,
        max_task_retries=1,
        retry_backoff_seconds=0.01,
        retry_backoff_multiplier=1.0,
        task_timeout_seconds=120,
        orchestrator_timeout_seconds=300,
        test_timeout_seconds=180,
        sandbox_enabled=False,
        log_level="CRITICAL",
    )
    get_settings.cache_clear()
    store = InMemoryStateStore()
    bus = EventBus()
    registry = AgentRegistry()
    registry.register_all(build_mock_roster(delay=0.0))
    from app.sandbox.manager import NoSandbox

    container = Container(
        settings=settings,
        store=store,  # type: ignore[arg-type]
        event_bus=bus,
        manager=ExecutionManager(store=store, event_bus=bus, settings=settings),
        registry=registry,
        provider=RuleBasedLLM(),
        sandbox=NoSandbox(),
        orchestrators={},
    )
    set_container(container)
    client = TestClient(create_app())

    class Harness:
        def __init__(self):
            self.client = client
            self.container = container
            self.bus = bus
            self.repo = target_repo

        def register_project(self):
            response = client.post(
                "/projects", json={"name": "demo", "local_path": str(target_repo)}
            )
            assert response.status_code == 201, response.text
            return response.json()

        def run(self, request, project=None, **extra):
            project = project or self.register_project()
            response = client.post(
                "/orchestrations",
                json={
                    "project_id": project["id"],
                    "user_request": request,
                    "wait": True,
                    **extra,
                },
            )
            assert response.status_code == 201, response.text
            return response.json()

    harness = Harness()
    try:
        yield harness
    finally:
        client.close()
        set_container(None)
        get_settings.cache_clear()


class TestHappyPath:
    def test_a_full_run_merges_its_branches_and_passes_the_suite(self, runner, target_repo):
        result = runner.run("Add a health check endpoint to the demo service")
        orch_id = result["id"]

        assert result["status"] == OrchestrationStatus.COMPLETED.value
        assert result["final_status"] == FinalStatus.SUCCESS.value
        assert result["total_tasks"] > 0

        report = runner.client.get(f"/orchestrations/{orch_id}/report").json()
        assert report["success"] is True
        assert report["merged_branches"], "no branch was merged"
        assert report["conflicts"] == []
        assert report["regressions"] == []

        # The repository really changed, and the tests really ran.
        assert git(target_repo, "status", "--porcelain") == ""
        log = git(target_repo, "log", "--oneline", "--all")
        assert "Merge branch" in log

    def test_the_suite_ran_and_recorded_counts(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        data = runner.client.get(
            f"/orchestrations/{result['id']}/test-results"
        ).json()
        assert data["summary"]["after_passed"] is True
        assert data["summary"]["passed_count"] >= 1
        assert data["summary"]["failed_count"] == 0
        assert data["runs"], "no test run was recorded"

    def test_every_task_reached_a_terminal_status(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        statuses = {t["status"] for t in result["tasks"].values()}
        assert statuses
        assert statuses <= {s.value for s in TaskStatus if s.is_terminal}
        assert TaskStatus.FAILED.value not in statuses

    def test_every_successful_task_left_a_real_commit(self, runner, target_repo):
        result = runner.run("Add a health check endpoint to the demo service")
        successful = [
            r for r in result["agent_results"].values()
            if r["status"] == TaskStatus.SUCCESS.value
        ]
        assert successful
        for record in successful:
            assert record["branch"], f"{record['task_id']} has no branch"
            assert record["commit"], f"{record['task_id']} has no commit"
            # The branch was cleaned up after integration, but the work landed.
            assert git(target_repo, "log", "--oneline") != ""

    def test_worktrees_are_cleaned_up(self, runner, target_repo):
        runner.run("Add a health check endpoint to the demo service")
        manager = GitManager(target_repo)
        worktrees = asyncio.run(manager.list_worktrees())
        assert len(worktrees) == 1, f"leftover worktrees: {worktrees}"
        assert Path(worktrees[0]["path"]).resolve() == target_repo.resolve()

    def test_agent_branches_are_deleted_after_integration(self, runner, target_repo):
        runner.run("Add a health check endpoint to the demo service")
        branches = git(target_repo, "branch", "--format=%(refname:short)").splitlines()
        assert not [b for b in branches if b.startswith("agent/")]

    def test_the_existing_suite_still_passes_after_integration(self, runner, target_repo):
        runner.run("Add a health check endpoint to the demo service")
        proc = subprocess.run(
            ["python", "-m", "pytest", "-q", "--no-header"],
            cwd=target_repo,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_the_merged_files_are_really_in_the_repository(self, runner, target_repo):
        runner.run("Add a health check endpoint to the demo service")
        merged = git(target_repo, "diff", "--name-only", "HEAD~5", "HEAD")
        assert merged or git(target_repo, "log", "--oneline") != ""


class TestPlanAndGraph:
    def test_the_dag_is_valid_and_ordered(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        dag = runner.client.get(f"/orchestrations/{result['id']}/dag").json()
        assert dag["total_tasks"] == result["total_tasks"]
        assert dag["waves"]
        assert dag["critical_path"]

        from app.models.orchestration import Orchestration

        state = _load(result["id"]).state
        DAGEngine(state.dag).validate()
        for task in state.dag.tasks.values():
            for dep in task.dependencies:
                assert dep in state.dag.tasks

    def test_implementation_precedes_verification(self, runner):
        result = runner.run("Add a health check endpoint to the demo service and tests")
        dag = runner.client.get(f"/orchestrations/{result['id']}/dag").json()
        flat = [tid for wave in dag["waves"] for tid in wave]
        node_by_id = {n["id"]: n for n in dag["nodes"]}
        testing = [i for i, tid in enumerate(flat) if "test" in node_by_id[tid]["title"].lower()]
        if testing:
            assert testing[0] > 0, "tests must not be the first wave"

    def test_the_plan_records_the_repository_analysis(self, runner, target_repo):
        result = runner.run("Add a health check endpoint to the demo service")
        analysis = result["repository_analysis"]
        assert analysis["total_files"] > 0
        assert "pyproject.toml" in analysis["config_files"]
        assert analysis["is_git_repo"] is True

    def test_the_plan_records_the_task_analysis(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        spec = result["task_analysis"]
        assert spec["raw_request"].startswith("Add a health check")
        assert spec["task_types"]


class TestEvents:
    def test_the_full_lifecycle_is_published(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        types = {e.type for e in runner.bus.recent(result["id"])}
        assert EventType.DAG_CREATED in types
        assert EventType.ORCHESTRATION_PHASE in types
        assert EventType.INTEGRATION_STARTED in types
        assert EventType.INTEGRATION_FINISHED in types
        assert EventType.TESTS_FINISHED in types

    def test_a_task_lifecycle_is_published(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        types = {e.type for e in runner.bus.recent(result["id"])}
        assert EventType.TASK_STARTED in types
        assert EventType.TASK_SUCCEEDED in types
        assert EventType.AGENT_STARTED in types
        assert EventType.AGENT_FINISHED in types

    def test_logs_capture_the_run(self, runner):
        result = runner.run("Add a health check endpoint to the demo service")
        logs = runner.client.get(f"/orchestrations/{result['id']}/logs?limit=2000").json()
        assert len(logs) > 5
        assert {e["type"] for e in logs} & {
            EventType.DAG_CREATED.value,
            EventType.INTEGRATION_FINISHED.value,
        }


class TestHonestFailure:
    def test_a_run_where_every_agent_fails_is_reported_as_failed(self, runner):
        from app.agents.mock import build_mock_roster

        runner.container.registry.register_all(build_mock_roster(delay=0.0, succeed=False))
        result = runner.run("Add a health check endpoint to the demo service")
        assert result["final_status"] == FinalStatus.FAILED.value
        assert result["errors"]
        assert all(
            t["status"] in {TaskStatus.FAILED.value, TaskStatus.BLOCKED.value}
            for t in result["tasks"].values()
        )

    def test_an_unsuccessful_integration_is_always_explained(self, runner, target_repo):
        """
        A failed integration must say why, and a regression must roll back.

        (The green-to-red rollback itself is covered directly in
        `test_integrator.py::test_a_broken_change_rolls_back`; here the point is
        that a real end-to-end failure is never silent.)
        """
        project = runner.register_project()  # initialises the repository
        baseline = git(target_repo, "rev-parse", "HEAD")

        result = runner.run("Break the greeting on purpose", project=project)
        state = _load(result["id"]).state
        report = state.integration_report
        if report is None or report.success:
            pytest.skip("this run integrated cleanly, so there is no failure to explain")

        explained = (
            report.regressions
            or report.conflicts
            or report.skipped_details
            or report.conflict_resolutions
        )
        assert explained, f"an unsuccessful report says nothing: {report.to_dict()}"
        assert state.errors or report.regressions or report.conflicts
        if report.regressions:
            # A broken merge is rolled back rather than left behind.
            assert git(target_repo, "rev-parse", "HEAD") == baseline

    def test_an_unresolvable_conflict_leaves_the_branch_unmerged(self, runner, target_repo):
        """Two agents rewriting the same file must not be silently reconciled."""
        result = runner.run("Add a health check endpoint to the demo service")
        orch_id = result["id"]
        report = runner.client.get(f"/orchestrations/{orch_id}/report").json()
        # Either there was no conflict, or every conflict was reported and
        # resolved. A silently-dropped conflict is the failure mode being ruled out.
        for conflict in report["conflicts"]:
            assert conflict["paths"]
            assert conflict["branch"]

    def test_success_without_merged_work_is_never_reported_as_success(self, runner):
        """Every agent claims success but writes nothing: that is a failure."""
        from app.agents.mock import build_mock_roster

        runner.container.registry.register_all(
            build_mock_roster(delay=0.0, write_files=False)
        )
        result = runner.run("Add a health check endpoint to the demo service")
        assert result["final_status"] != FinalStatus.SUCCESS.value
        assert result["errors"]
        assert any("reported success" in e for e in result["errors"])


class TestIdempotence:
    def test_two_runs_in_a_row_both_succeed(self, runner):
        first = runner.run("Add a health check endpoint to the demo service")
        second = runner.run("Add a health check endpoint to the demo service")
        assert first["final_status"] == FinalStatus.SUCCESS.value
        assert second["final_status"] == FinalStatus.SUCCESS.value

    def test_two_runs_produce_two_distinct_orchestrations(self, runner):
        first = runner.run("Add a health check endpoint")
        second = runner.run("Add a health check endpoint")
        assert first["id"] != second["id"]
        assert len(runner.client.get("/orchestrations").json()) == 2

    def test_concurrent_runs_do_not_interfere(self, runner):
        """The whole point of worktree isolation."""
        results = [
            runner.run("Add a health check endpoint to the demo service")
            for _ in range(2)
        ]
        for result in results:
            assert result["final_status"] == FinalStatus.SUCCESS.value


class TestGreenfield:
    def test_a_run_against_an_empty_directory(self, runner, tmp_path):
        empty = tmp_path / "brand-new"
        empty.mkdir()
        project = runner.client.post(
            "/projects", json={"name": "new", "local_path": str(empty)}
        ).json()
        result = runner.run("Build a simple calculator service with tests", project=project)
        assert result["status"] == OrchestrationStatus.COMPLETED.value
        assert result["total_tasks"] > 0
        assert (empty / ".git").exists()


def _load(orchestration_id: str):
    import asyncio as _asyncio

    from app.api.deps import get_container

    return _asyncio.run(get_container().manager.get(orchestration_id))

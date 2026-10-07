"""
Integrator tests.

The integrator is where an agent's work becomes a real repository state, so
these tests run real git and, where cheap, a real test suite. The rule that
matters most is asserted repeatedly: a conflict is never silently resolved by
picking a winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.events import EventBus
from app.git.manager import GitManager, is_git_installed
from app.integrator.conflict_resolver import (
    ConflictRegion,
    ConflictResolver,
    GitMarkers,
    ResolutionReport,
    summarise_diff,
)
from app.integrator.system import (
    ChangeCollector,
    ChangeRecord,
    FinalReviewer,
    Integrator,
    RegressionDetector,
)
from app.integrator.test_runner import (
    TestRunner,
    _has_python_tests,
    split_command,
    sys_executable,
)
from app.llm.mock import MockLLMProvider, ScriptedLLM
from app.models.domain import (
    AgentResult,
    DAG,
    ExecutionState,
    FinalStatus,
    Task,
    TaskStatus,
    TaskType,
    TestRunResult,
    new_id,
)

pytestmark = pytest.mark.skipif(
    not is_git_installed(), reason="git is not installed on PATH"
)


def run_git(cwd: Path, *args: str, check: bool = True) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    ).stdout.strip()


def markers(ours: str, theirs: str, base: str = "") -> str:
    block = f"<<<<<<< HEAD\n{ours}\n"
    if base:
        block += f"||||||| {base}\n{base}\n"
    block += f"=======\n{theirs}\n>>>>>>> feature\n"
    return block


# ── command detection ───────────────────────────────────────────────────────


class TestDetectCommand:
    def test_pyproject_wins(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        assert TestRunner().detect_command(tmp_path).kind == "pytest"

    def test_tests_directory_triggers_pytest(self, tmp_path):
        (tmp_path / "tests").mkdir()
        assert TestRunner().detect_command(tmp_path).kind == "pytest"

    def test_bare_test_file_triggers_pytest(self, tmp_path):
        (tmp_path / "test_thing.py").write_text("", encoding="utf-8")
        assert TestRunner().detect_command(tmp_path).kind == "pytest"

    def test_nested_test_file_triggers_pytest(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "thing_test.py").write_text("", encoding="utf-8")
        assert TestRunner().detect_command(tmp_path).kind == "pytest"

    def test_npm_test_script(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest run"}}), encoding="utf-8"
        )
        command = TestRunner().detect_command(tmp_path)
        assert command.kind == "npm"
        assert "test" in command.argv

    def test_npm_without_a_test_script_needs_node_modules(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"build": "tsc"}}), encoding="utf-8"
        )
        assert TestRunner().detect_command(tmp_path) is None
        (tmp_path / "node_modules").mkdir()
        assert TestRunner().detect_command(tmp_path).kind == "npm"

    def test_malformed_package_json_falls_through(self, tmp_path):
        (tmp_path / "package.json").write_text("{not json", encoding="utf-8")
        assert TestRunner().detect_command(tmp_path) is None

    def test_go_and_cargo(self, tmp_path):
        go = tmp_path / "go"
        go.mkdir()
        (go / "go.mod").write_text("module x\n", encoding="utf-8")
        assert TestRunner().detect_command(go).kind == "go"
        cargo = tmp_path / "rs"
        cargo.mkdir()
        (cargo / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        assert TestRunner().detect_command(cargo).kind == "cargo"

    def test_makefile_test_target(self, tmp_path):
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")
        assert TestRunner().detect_command(tmp_path).kind == "make"

    def test_makefile_without_a_test_target(self, tmp_path):
        (tmp_path / "Makefile").write_text("build:\n\techo hi\n", encoding="utf-8")
        assert TestRunner().detect_command(tmp_path) is None

    def test_explicit_override_wins_over_detection(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        runner = TestRunner(command="pytest -x --tb=short")
        command = runner.detect_command(tmp_path)
        assert command.kind == "custom"
        assert command.argv == ["pytest", "-x", "--tb=short"]

    def test_missing_directory_has_no_command(self, tmp_path):
        assert TestRunner().detect_command(tmp_path / "nope") is None
        assert TestRunner().can_run(tmp_path / "nope") is False

    def test_command_renders_readably(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        assert "-m pytest" in str(TestRunner().detect_command(tmp_path))

    def test_sys_executable_is_usable(self):
        assert sys_executable()

    def test_has_python_tests(self, tmp_path):
        (tmp_path / "test_x.py").write_text("", encoding="utf-8")
        assert _has_python_tests(tmp_path) is True


# ── running the suite ───────────────────────────────────────────────────────


@pytest.fixture
def green_repo(tmp_path: Path) -> Path:
    root = tmp_path / "green"
    root.mkdir()
    (root / "test_ok.py").write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def red_repo(tmp_path: Path) -> Path:
    root = tmp_path / "red"
    root.mkdir()
    (root / "test_bad.py").write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert False\n",
        encoding="utf-8",
    )
    return root


class TestRun:
    @pytest.mark.asyncio
    async def test_green_suite_passes_and_counts(self, green_repo):
        result = await TestRunner(timeout=120).run(green_repo)
        assert result.passed is True
        assert result.passed_count == 2
        assert result.failed_count == 0
        assert result.exit_code == 0
        assert result.skipped is False

    @pytest.mark.asyncio
    async def test_red_suite_fails_and_counts(self, red_repo):
        result = await TestRunner(timeout=120).run(red_repo)
        assert result.passed is False
        assert result.failed_count == 1
        assert result.passed_count == 1
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_a_repository_with_no_tests_is_skipped_not_failed(self, tmp_path):
        root = tmp_path / "bare"
        root.mkdir()
        result = await TestRunner().run(root)
        assert result.skipped is True
        assert result.passed is True  # nothing to fail
        assert "nothing to run" in result.stdout.lower()

    @pytest.mark.asyncio
    async def test_missing_binary_is_a_failure_not_an_exception(self, tmp_path):
        root = tmp_path / "custom"
        root.mkdir()
        result = await TestRunner(command="definitely-not-a-real-binary").run(root)
        assert result.passed is False
        assert result.exit_code == 127
        assert "not found" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout_kills_the_suite(self, green_repo):
        result = await TestRunner(
            command=f'"{sys_executable()}" -c "import time; time.sleep(30)"'
        ).run(green_repo, timeout=1.0)
        assert result.passed is False
        assert result.exit_code == 124
        assert "killed" in result.stderr

    def test_a_windows_style_path_survives_command_splitting(self):
        """A backslash is a path separator, not an escape character."""
        argv = split_command("C:\\Python314\\python.exe -m pytest -q")
        assert argv[0] == "C:\\Python314\\python.exe"
        assert argv[1:] == ["-m", "pytest", "-q"]

    def test_quoted_arguments_with_spaces_are_kept_together(self):
        assert split_command('pytest -k "a test"')[2] == "a test"

    @pytest.mark.asyncio
    async def test_secrets_are_scrubbed_from_the_environment(self, green_repo, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-should-not-leak")
        env = TestRunner._env()
        assert "OPENAI_API_KEY" not in env
        assert "GITHUB_TOKEN" not in env
        assert env["CI"] == "1"
        assert env["NO_COLOR"] == "1"

    def test_counts_are_parsed_from_pytest_output(self):
        result = TestRunResult(command="pytest", passed=True, stdout="2 passed, 1 warning")
        TestRunner._parse_counts(result)
        assert (result.passed_count, result.failed_count) == (2, 0)

    def test_errors_count_as_failures(self):
        result = TestRunResult(
            command="pytest", passed=False, stdout="1 failed, 2 errors, 3 passed"
        )
        TestRunner._parse_counts(result)
        assert result.failed_count == 3
        assert result.passed_count == 3

    def test_non_pytest_output_falls_back_to_line_counting(self):
        result = TestRunResult(
            command="go test", passed=False, stdout="ok  x\nok  y\nFAIL  z\n"
        )
        TestRunner._parse_counts(result)
        assert (result.passed_count, result.failed_count) == (2, 1)

    def test_output_with_no_counts_stays_zero(self):
        result = TestRunResult(command="x", passed=True, stdout="nothing numeric")
        TestRunner._parse_counts(result)
        assert (result.passed_count, result.failed_count) == (0, 0)

    @pytest.mark.asyncio
    async def test_run_with_retry_stops_on_success(self, green_repo):
        results = await TestRunner(timeout=120).run_with_retry(str(green_repo), attempts=3)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_with_retry_retries_a_failure(self, red_repo):
        results = await TestRunner(timeout=120).run_with_retry(str(red_repo), attempts=2)
        assert len(results) == 2
        assert all(not r.passed for r in results)

    @pytest.mark.asyncio
    async def test_sandbox_path_is_used_when_configured(self, green_repo):
        seen = {}

        class FakeSandbox:
            async def run(self, repo, argv, timeout=None):
                seen["argv"] = argv
                return {"exit_code": 0, "stdout": "5 passed", "stderr": ""}

        runner = TestRunner(
            use_sandbox=True, sandbox_runner=FakeSandbox(), timeout=60
        )
        result = await runner.run(green_repo)
        assert seen["argv"][-1] == "--no-header"
        assert result.passed is True
        assert result.passed_count == 5

    @pytest.mark.asyncio
    async def test_sandbox_failure_is_reported_not_raised(self, green_repo):
        class BrokenSandbox:
            async def run(self, repo, argv, timeout=None):
                raise RuntimeError("docker daemon unreachable")

        runner = TestRunner(use_sandbox=True, sandbox_runner=BrokenSandbox())
        result = await runner.run(green_repo)
        assert result.passed is False
        assert "docker daemon unreachable" in result.stderr


# ── change collection ───────────────────────────────────────────────────────


def make_state(tasks: list[Task], results: dict[str, AgentResult]) -> ExecutionState:
    state = ExecutionState(
        project_id="test-project",
        original_task="implement the requested feature",
        repository=str(Path.cwd()),
    )
    for task in tasks:
        state.tasks[task.id] = task
    state.dag = DAG(tasks={t.id: t for t in tasks})
    state.dag.rebuild_edges()
    state.agent_results.update(results)
    return state


class TestChangeCollector:
    @pytest.mark.asyncio
    async def test_successful_results_with_a_branch_are_mergeable(self, git):
        task = Task(title="a", type=TaskType.BACKEND, branch="agent/a")
        state = make_state(
            [task],
            {
                task.id: AgentResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    summary="ok",
                    branch="agent/a",
                    commit="a" * 40,
                )
            },
        )
        run_git(Path(git.repo_path), "branch", "agent/a")
        mergeable, skipped = await ChangeCollector().collect(state, git)
        assert [r.task_id for r in mergeable] == [task.id]
        assert skipped == []

    @pytest.mark.asyncio
    async def test_failed_tasks_are_skipped_with_a_reason(self, git):
        task = Task(title="a", type=TaskType.BACKEND, branch="agent/a")
        state = make_state(
            [task],
            {
                task.id: AgentResult(
                    task_id=task.id, status=TaskStatus.FAILED, summary="broke", branch="agent/a"
                )
            },
        )
        mergeable, skipped = await ChangeCollector().collect(state, git)
        assert mergeable == []
        assert "FAILED" in skipped[0].skip_reason

    @pytest.mark.asyncio
    async def test_a_successful_task_without_a_branch_is_skipped_with_a_reason(self, git):
        task = Task(title="a", type=TaskType.BACKEND)
        state = make_state(
            [task],
            {task.id: AgentResult(task_id=task.id, status=TaskStatus.SUCCESS, summary="ok")},
        )
        _, skipped = await ChangeCollector().collect(state, git)
        assert skipped[0].status == "SUCCESS"
        assert "no branch" in skipped[0].skip_reason

    @pytest.mark.asyncio
    async def test_a_missing_branch_is_skipped_with_a_reason(self, git):
        task = Task(title="a", type=TaskType.BACKEND, branch="agent/gone")
        state = make_state(
            [task],
            {
                task.id: AgentResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    summary="ok",
                    branch="agent/gone",
                    files_changed=["a.py"],
                )
            },
        )
        _, skipped = await ChangeCollector().collect(state, git)
        assert "no longer exists" in skipped[0].skip_reason

    @pytest.mark.asyncio
    async def test_a_success_with_no_changes_is_skipped(self, git):
        task = Task(title="a", type=TaskType.BACKEND, branch="agent/a")
        run_git(Path(git.repo_path), "branch", "agent/a")
        state = make_state(
            [task],
            {
                task.id: AgentResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    summary="did nothing",
                    branch="agent/a",
                )
            },
        )
        _, skipped = await ChangeCollector().collect(state, git)
        assert "no file changes" in skipped[0].skip_reason

    @pytest.mark.asyncio
    async def test_collect_is_ordered_topologically(self, git):
        first = Task(title="first", type=TaskType.BACKEND, branch="agent/a")
        second = Task(
            title="second", type=TaskType.TESTING, branch="agent/b", dependencies=[first.id]
        )
        run_git(Path(git.repo_path), "branch", "agent/a")
        run_git(Path(git.repo_path), "branch", "agent/b")
        results = {
            t.id: AgentResult(
                task_id=t.id,
                status=TaskStatus.SUCCESS,
                summary="ok",
                branch=t.branch,
                files_changed=["x.py"],
            )
            for t in (first, second)
        }
        mergeable, _ = await ChangeCollector().collect(make_state([second, first], results), git)
        assert [r.task_id for r in mergeable] == [first.id, second.id]

    @pytest.mark.asyncio
    async def test_tasks_without_results_are_ignored(self, git):
        state = make_state([Task(title="a", type=TaskType.BACKEND)], {})
        mergeable, skipped = await ChangeCollector().collect(state, git)
        assert (mergeable, skipped) == ([], [])

    def test_change_record_serialises(self):
        record = ChangeRecord(task_id="t", title="x", branch="b", commit="c")
        data = record.to_dict()
        assert data["task_id"] == "t"
        assert data["files_changed"] == []
        json.dumps(data)


# ── regression detection ────────────────────────────────────────────────────


class TestRegressionDetector:
    def test_clean_pass_after_a_clean_pass_is_not_a_regression(self):
        before = TestRunResult(command="p", passed=True, passed_count=5, failed_count=0)
        after = TestRunResult(command="p", passed=True, passed_count=6, failed_count=0)
        assert RegressionDetector().compare(before, after) == []

    def test_a_new_failure_after_a_green_baseline_is_a_regression(self):
        before = TestRunResult(command="p", passed=True, passed_count=5)
        after = TestRunResult(
            command="p", passed=False, exit_code=1, passed_count=4, failed_count=1
        )
        findings = RegressionDetector().compare(before, after)
        assert any("passed before integration" in f for f in findings)

    def test_more_failures_than_baseline_is_a_regression(self):
        before = TestRunResult(command="p", passed=False, passed_count=5, failed_count=1)
        after = TestRunResult(command="p", passed=False, passed_count=3, failed_count=4)
        findings = RegressionDetector().compare(before, after)
        assert any("increased by 3" in f for f in findings)

    def test_fewer_passing_tests_is_flagged_as_coverage_loss(self):
        before = TestRunResult(command="p", passed=True, passed_count=10)
        after = TestRunResult(command="p", passed=True, passed_count=7)
        findings = RegressionDetector().compare(before, after)
        assert any("dropped from 10 to 7" in f for f in findings)

    def test_an_improvement_is_not_a_regression(self):
        before = TestRunResult(command="p", passed=False, passed_count=2, failed_count=5)
        after = TestRunResult(command="p", passed=True, passed_count=7, failed_count=0)
        assert RegressionDetector().compare(before, after) == []

    def test_no_baseline_means_no_verdict(self):
        after = TestRunResult(command="p", passed=False, failed_count=3)
        assert RegressionDetector().compare(None, after) == []

    def test_a_skipped_run_means_no_verdict(self):
        before = TestRunResult(command="", passed=True, skipped=True)
        after = TestRunResult(command="p", passed=False, failed_count=3)
        assert RegressionDetector().compare(before, after) == []
        assert RegressionDetector().compare(
            before, TestRunResult(command="p", passed=False, skipped=True)
        ) == []


# ── conflict resolution ─────────────────────────────────────────────────────


class TestGitMarkers:
    def test_split_extracts_both_sides(self):
        ours, theirs = GitMarkers.split(markers("ours line", "theirs line"))
        assert ours == "ours line"
        assert theirs == "theirs line"

    def test_split_handles_a_three_way_block(self):
        block = "<<<<<<< HEAD\nours\n||||||| base\noriginal\n=======\ntheirs\n>>>>>>> x\n"
        assert GitMarkers.split(block) == ("ours", "theirs")

    def test_split_finds_every_hunk(self):
        ours, theirs = GitMarkers.split(
            markers("o1", "t1") + "middle\n" + markers("o2", "t2")
        )
        assert ours == "o1\no2"
        assert theirs == "t1\nt2"

    def test_has_markers(self):
        assert GitMarkers.has_markers(markers("a", "b")) is True
        assert GitMarkers.has_markers("clean content\n") is False
        assert GitMarkers.has_markers("text <<<<<<< not at line start\n") is False

    def test_split_on_clean_text_is_empty(self):
        assert GitMarkers.split("nothing here") == ("", "")

    def test_split_full_returns_the_base_of_a_three_way_block(self):
        block = "<<<<<<< HEAD\nours\n||||||| base label\noriginal\n=======\ntheirs\n>>>>>>> x\n"
        assert GitMarkers.split_full(block) == ("ours", "theirs", "original")

    def test_split_full_gives_an_empty_base_for_a_two_way_block(self):
        assert GitMarkers.split_full(markers("o", "t")) == ("o", "t", "")


class TestParseRegions:
    def test_a_single_region_is_parsed(self):
        detail = {"path": "a.py", "raw": "before\n" + markers("ours", "theirs") + "\nafter\n"}
        regions = ConflictResolver._parse_regions("a.py", detail)
        assert len(regions) == 1
        assert regions[0].ours == "ours"
        assert regions[0].theirs == "theirs"
        assert regions[0].path == "a.py"

    def test_multiple_regions_are_parsed(self):
        detail = {"path": "a.py", "raw": markers("o1", "t1") + markers("o2", "t2")}
        regions = ConflictResolver._parse_regions("a.py", detail)
        assert len(regions) == 2
        assert [r.ours for r in regions] == ["o1", "o2"]

    def test_three_way_blocks_are_parsed(self):
        detail = {"path": "a.py", "raw": markers("ours", "theirs", base="original")}
        regions = ConflictResolver._parse_regions("a.py", detail)
        assert len(regions) == 1
        assert regions[0].base == "original"

    def test_no_markers_yields_no_regions(self):
        assert ConflictResolver._parse_regions("a.py", {"raw": "clean\n"}) == []

    def test_missing_raw_yields_no_regions(self):
        assert ConflictResolver._parse_regions("a.py", {}) == []

    def test_is_deletion(self):
        assert ConflictRegion("p", "", "x").is_deletion is True
        assert ConflictRegion("p", "x", "y").is_deletion is False


class TestConflictResolver:
    @pytest.mark.asyncio
    async def test_identical_sides_resolve_structurally(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("same", "same")}]
        )
        assert report.success is True
        assert report.strategies["a.py"] == "structural"
        assert report.resolved_contents["a.py"].strip() == "same"

    @pytest.mark.asyncio
    async def test_a_superset_on_the_incoming_side_is_kept(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("a\nb", "a\nb\nc")}]
        )
        assert report.success is True
        assert report.resolved_contents["a.py"].strip() == "a\nb\nc"

    @pytest.mark.asyncio
    async def test_a_superset_on_the_current_side_is_kept(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("a\nb\nc", "a\nb")}]
        )
        assert report.resolved_contents["a.py"].strip() == "a\nb\nc"

    @pytest.mark.asyncio
    async def test_disjoint_additions_on_a_shared_base_are_interleaved(self):
        """Two agents appending different helpers to the same untouched base."""
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".",
            [
                {
                    "path": "a.py",
                    "raw": markers(
                        "import os\ndef ours():\n    pass",
                        "import os\ndef theirs():\n    pass",
                        base="import os",
                    ),
                }
            ],
        )
        assert report.success is True
        merged = report.resolved_contents["a.py"]
        assert "def ours()" in merged
        assert "def theirs()" in merged

    @pytest.mark.asyncio
    async def test_without_a_base_two_rewrites_are_never_interleaved(self):
        """Concatenating `return 1` and `return 2` is not code."""
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("return 1", "return 2")}]
        )
        assert report.success is False
        assert report.resolved_contents == {}

    @pytest.mark.asyncio
    async def test_a_deletion_keeps_the_surviving_side(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("keep me", "")}]
        )
        assert report.success is True
        assert report.resolved_contents["a.py"].strip() == "keep me"

    @pytest.mark.asyncio
    async def test_a_genuine_conflict_is_reported_not_guessed(self):
        """Overlapping rewrites must fail loudly, never pick a winner."""
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("return 1", "return 2")}]
        )
        assert report.success is False
        assert report.unresolved == ["a.py"]
        assert "a.py" not in report.resolved_contents
        assert report.notes

    @pytest.mark.asyncio
    async def test_a_rename_style_conflict_is_not_silently_resolved(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": markers("def old_name():", "def new_name():")}]
        )
        assert report.success is False

    @pytest.mark.asyncio
    async def test_a_detail_without_markers_keeps_the_raw_content(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".", [{"path": "a.py", "raw": "already fine\n"}]
        )
        assert report.strategies["a.py"] == "raw"
        assert report.resolved_contents["a.py"] == "already fine\n"

    @pytest.mark.asyncio
    async def test_a_detail_without_a_path_is_skipped(self):
        report = await ConflictResolver().resolve_with_report(".", [{"raw": markers("a", "b")}])
        assert report.resolved == []
        assert report.success is True

    @pytest.mark.asyncio
    async def test_multiple_files_are_resolved_independently(self):
        resolver = ConflictResolver()
        report = await resolver.resolve_with_report(
            ".",
            [
                {"path": "ok.py", "raw": markers("same", "same")},
                {"path": "bad.py", "raw": markers("return 1", "return 2")},
            ],
        )
        assert report.resolved == ["ok.py"]
        assert report.unresolved == ["bad.py"]
        assert report.success is False

    @pytest.mark.asyncio
    async def test_an_llm_resolution_is_used_and_labelled(self, tmp_path):
        target = tmp_path / "a.py"
        target.write_text(markers("return 1", "return 2"), encoding="utf-8")
        provider = ScriptedLLM(responses=['<file path="a.py">return 3\n</file>'])
        resolver = ConflictResolver(llm=provider)
        report = await resolver.resolve_with_report(str(tmp_path), [{"path": "a.py"}])
        assert report.success is True
        assert report.strategies["a.py"] == "llm"
        assert report.resolved_contents["a.py"].strip() == "return 3"

    @pytest.mark.asyncio
    async def test_a_failed_llm_falls_back_to_structural(self, tmp_path):
        resolver = ConflictResolver(llm=MockLLMProvider(response="I refuse"))
        report = await resolver.resolve_with_report(
            str(tmp_path), [{"path": "a.py", "raw": markers("a", "a")}]
        )
        assert report.strategies["a.py"] == "structural"

    @pytest.mark.asyncio
    async def test_a_failed_llm_does_not_invent_a_resolution(self, tmp_path):
        resolver = ConflictResolver(llm=MockLLMProvider(response="I refuse"))
        report = await resolver.resolve_with_report(
            str(tmp_path), [{"path": "a.py", "raw": markers("return 1", "return 2")}]
        )
        assert report.success is False

    @pytest.mark.asyncio
    async def test_resolve_returns_only_resolved_contents(self, tmp_path):
        resolver = ConflictResolver()
        contents = await resolver.resolve(
            ".",
            [
                {"path": "ok.py", "raw": markers("same", "same")},
                {"path": "bad.py", "raw": markers("return 1", "return 2")},
            ],
        )
        assert set(contents) == {"ok.py"}

    @pytest.mark.asyncio
    async def test_the_prompt_shows_both_sides_and_the_ancestor(self, tmp_path):
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        provider = ScriptedLLM(responses=['<file path="a.py">merged\n</file>'])
        await ConflictResolver(llm=provider).resolve_with_report(
            str(tmp_path), [{"path": "a.py", "raw": markers("return 1", "return 2", base="return 0")}]
        )
        prompt = provider.prompts[-1]
        assert "return 1" in prompt
        assert "return 2" in prompt
        assert "return 0" in prompt

    @pytest.mark.asyncio
    async def test_empty_input_is_trivially_successful(self):
        report = await ConflictResolver().resolve_with_report(".", [])
        assert report.success is True
        assert report.resolved_contents == {}

    def test_report_defaults(self):
        report = ResolutionReport()
        assert report.success is True
        json.dumps(report.to_dict())


def test_summarise_diff():
    diff = summarise_diff("a\nb\n", "a\nc\n", "f.py")
    assert "-b" in diff
    assert "+c" in diff
    assert "a/f.py" in diff and "b/f.py" in diff


# ── final review ────────────────────────────────────────────────────────────


class TestFinalReviewer:
    @pytest.mark.asyncio
    async def test_no_llm_means_no_findings(self):
        assert await FinalReviewer().review("diff --git a/x b/x") == []

    @pytest.mark.asyncio
    async def test_an_empty_diff_is_not_reviewed(self):
        provider = MockLLMProvider()
        assert await FinalReviewer(llm=provider).review("   ") == []
        assert provider.prompts == []

    @pytest.mark.asyncio
    async def test_findings_are_parsed_from_json(self):
        provider = ScriptedLLM(
            responses=[
                '{"findings": [{"severity": "critical", "file": "a.py", "line": 3, '
                '"issue": "sql injection", "fix": "parameterise"}]}'
            ]
        )
        findings = await FinalReviewer(llm=provider).review("diff")
        assert findings[0]["severity"] == "critical"
        assert findings[0]["issue"] == "sql injection"

    @pytest.mark.asyncio
    async def test_non_dict_findings_are_dropped(self):
        provider = ScriptedLLM(
            responses=['{"findings": ["a string", {"severity": "minor"}]}']
        )
        findings = await FinalReviewer(llm=provider).review("diff")
        assert findings == [{"severity": "minor"}]

    @pytest.mark.asyncio
    async def test_a_prose_review_is_kept_as_an_info_finding(self):
        findings = await FinalReviewer(llm=MockLLMProvider(response="looks fine")).review("diff")
        assert findings[0]["severity"] == "info"
        assert findings[0]["issue"] == "looks fine"

    @pytest.mark.asyncio
    async def test_provider_failure_is_swallowed(self):
        provider = MockLLMProvider()
        provider.fail_times = 99
        provider.max_retries = 0
        assert await FinalReviewer(llm=provider).review("diff") == []


# ── the full integration ────────────────────────────────────────────────────


@pytest.fixture
def integration_repo(tmp_path: Path) -> Path:
    root = tmp_path / "integrated"
    root.mkdir()
    (root / "app.py").write_text("def main():\n    return 'base'\n", encoding="utf-8")
    (root / "test_app.py").write_text(
        "from app import main\n\n\ndef test_main():\n    assert main() is not None\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
async def integrated_git(integration_repo):
    manager = GitManager(integration_repo)
    await manager.init_repo()
    await manager.commit_all(integration_repo, "base")
    return manager


async def agent_branch(
    git: GitManager, repo: Path, name: str, filename: str, content: str
) -> str:
    """Simulate one agent: its own worktree, its own branch, its own commit."""
    path = await git.create_worktree(f"agent/{name}", repo.parent / f"wt-{name}")
    (path / filename).write_text(content, encoding="utf-8")
    await git.commit_all(path, f"{name} work")
    return f"agent/{name}"


def state_with(branch: str, status: TaskStatus = TaskStatus.SUCCESS) -> ExecutionState:
    task = Task(title="implement feature", type=TaskType.BACKEND, branch=branch)
    state = make_state(
        [task],
        {
            task.id: AgentResult(
                task_id=task.id,
                status=status,
                summary="done",
                branch=branch,
                commit="c" * 40,
                files_changed=["feature.py"],
            )
        },
    )
    return state


class TestIntegrator:
    @pytest.mark.asyncio
    async def test_a_clean_merge_passes_the_suite(self, integrated_git, integration_repo, settings):
        branch = await agent_branch(
            integrated_git, integration_repo, "feature", "feature.py", "VALUE = 1\n"
        )
        state = state_with(branch)
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)
        assert report.success is True
        assert report.merged_branches == [branch]
        assert (integration_repo / "feature.py").exists()
        assert state.final_status is FinalStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_a_broken_change_rolls_back(self, integrated_git, integration_repo, settings):
        baseline = await integrated_git.snapshot_ref()
        branch = await agent_branch(
            integrated_git,
            integration_repo,
            "broken",
            "app.py",
            "def main():\n    return None\n",
        )
        # The branch breaks the existing test.
        (integration_repo / "app.py").write_text(
            "def main():\n    return 'base'\n", encoding="utf-8"
        )
        state = state_with(branch)
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)
        assert report.success is False
        assert report.regressions
        assert state.final_status is FinalStatus.FAILED
        assert await integrated_git.head_commit() == baseline
        assert not (integration_repo / "feature.py").exists()

    @pytest.mark.asyncio
    async def test_a_conflict_is_resolved_by_an_agent_and_merged(
        self, integrated_git, integration_repo, settings
    ):
        root = integration_repo
        (root / "shared.py").write_text("VALUE = 0\n", encoding="utf-8")
        await integrated_git.commit_all(root, "shared base")
        run_git(root, "checkout", "-b", "agent/conflict")
        (root / "shared.py").write_text("VALUE = 2\n", encoding="utf-8")
        await integrated_git.commit_all(root, "theirs")
        run_git(root, "checkout", "main")
        (root / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
        await integrated_git.commit_all(root, "ours")

        state = state_with("agent/conflict")
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
            llm=ScriptedLLM(
                responses=['<file path="shared.py">VALUE = 3\n</file>'],
                default='<file path="shared.py">VALUE = 3\n</file>',
            ),
        ).integrate(state)
        assert report.conflicts
        assert report.conflict_resolutions
        assert report.conflict_resolutions[0]["strategies"] == {"shared.py": "llm"}
        # The merge agent's result is what lands: this is the whole point.
        assert (root / "shared.py").read_text(encoding="utf-8").strip() == "VALUE = 3"
        assert report.merged_branches == ["agent/conflict"]
        assert report.success is True

    @pytest.mark.asyncio
    async def test_an_unresolvable_conflict_is_reported_and_not_merged(
        self, integrated_git, integration_repo, settings
    ):
        root = integration_repo
        (root / "clash.py").write_text("def value():\n    return 0\n", encoding="utf-8")
        await integrated_git.commit_all(root, "base")
        run_git(root, "checkout", "-b", "agent/clash")
        (root / "clash.py").write_text("def value():\n    return 2\n", encoding="utf-8")
        await integrated_git.commit_all(root, "theirs")
        run_git(root, "checkout", "main")
        (root / "clash.py").write_text("def value():\n    return 1\n", encoding="utf-8")
        await integrated_git.commit_all(root, "ours")

        state = state_with("agent/clash")
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)
        assert report.success is False
        assert report.merged_branches == []
        assert state.errors
        assert any("conflict" in e.lower() for e in state.errors)

    @pytest.mark.asyncio
    async def test_success_without_a_branch_is_not_reported_as_success(
        self, integrated_git, integration_repo, settings
    ):
        task = Task(title="did something", type=TaskType.BACKEND)
        state = make_state(
            [task],
            {
                task.id: AgentResult(
                    task_id=task.id,
                    status=TaskStatus.SUCCESS,
                    summary="ok",
                    files_changed=["x.py"],
                )
            },
        )
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)
        assert report.success is False
        assert report.skipped_branches
        assert report.skipped_details[0]["skip_reason"]
        assert any("reported success" in e for e in state.errors)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_type", [TaskType.ANALYSIS, TaskType.REVIEW])
    async def test_a_read_only_task_that_changes_nothing_does_not_downgrade_the_run(
        self, integrated_git, integration_repo, settings, task_type
    ):
        """
        An analyst and a reviewer that commit nothing did their job. Charging
        them as no-op successes made every well-formed plan report INCOMPLETE,
        because the analysis and review waves always end this way.
        """
        readonly = Task(title="read the repo", type=task_type)
        branch = await agent_branch(
            integrated_git, integration_repo, f"real-{task_type.value}", "feature.py", "V = 1\n"
        )
        real = Task(title="implement it", type=TaskType.BACKEND, branch=branch)

        state = make_state(
            [readonly, real],
            {
                readonly.id: AgentResult(
                    task_id=readonly.id,
                    status=TaskStatus.SUCCESS,
                    summary="analysed",
                ),
                real.id: AgentResult(
                    task_id=real.id,
                    status=TaskStatus.SUCCESS,
                    summary="wrote it",
                    branch=branch,
                    commit="c" * 40,
                    files_changed=["feature.py"],
                ),
            },
        )
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)

        assert report.success is True
        assert report.merged_branches == [branch]
        skipped_types = {d["task_type"] for d in report.skipped_details}
        assert skipped_types == {task_type.value}
        assert not any("reported success" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_nothing_to_merge_is_a_failure_not_a_success(
        self, integrated_git, settings
    ):
        state = make_state([], {})
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)
        assert report.success is False
        assert report.merged_branches == []

    @pytest.mark.asyncio
    async def test_a_succeeded_no_op_branch_is_deleted_but_a_failed_one_is_kept(
        self, integrated_git, integration_repo, settings
    ):
        """
        A read-only agent that commits nothing still leaves a branch behind.
        Those hold no evidence, so they are cleaned up; a failed task's branch
        is kept because its workspace is kept for inspection.
        """
        merged = await agent_branch(
            integrated_git, integration_repo, "merged", "feature.py", "V = 1\n"
        )
        noop = await agent_branch(
            integrated_git, integration_repo, "noop", "notes.md", "nothing\n"
        )
        kept = await agent_branch(
            integrated_git, integration_repo, "kept", "broken.py", "raise SystemExit(1)\n"
        )

        ok = Task(title="just read", type=TaskType.ANALYSIS, branch=noop)
        bad = Task(title="blew up", type=TaskType.BACKEND, branch=kept)
        good = Task(title="implement feature", type=TaskType.BACKEND, branch=merged)
        state = make_state(
            [ok, bad, good],
            {
                ok.id: AgentResult(
                    task_id=ok.id, status=TaskStatus.SUCCESS, summary="read it", branch=noop
                ),
                bad.id: AgentResult(
                    task_id=bad.id,
                    status=TaskStatus.FAILED,
                    summary="boom",
                    branch=kept,
                    commit="c" * 40,
                    files_changed=["broken.py"],
                ),
                good.id: AgentResult(
                    task_id=good.id,
                    status=TaskStatus.SUCCESS,
                    summary="done",
                    branch=merged,
                    commit="c" * 40,
                    files_changed=["feature.py"],
                ),
            },
        )

        await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)

        assert await integrated_git.branch_exists(merged) is False
        assert await integrated_git.branch_exists(noop) is False
        assert await integrated_git.branch_exists(kept) is True

    @pytest.mark.asyncio
    async def test_merged_branches_are_deleted_after_integration(
        self, integrated_git, integration_repo, settings
    ):
        branch = await agent_branch(
            integrated_git, integration_repo, "temp", "feature.py", "X = 1\n"
        )
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state_with(branch))
        assert report.success is True
        assert await integrated_git.branch_exists(branch) is False

    @pytest.mark.asyncio
    async def test_test_results_are_recorded_on_the_state(
        self, integrated_git, integration_repo, settings
    ):
        branch = await agent_branch(
            integrated_git, integration_repo, "recorded", "feature.py", "Y = 2\n"
        )
        state = state_with(branch)
        await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state)
        assert state.test_results["after_passed"] is True
        assert state.test_results["passed_count"] >= 1
        assert state.test_results["failures"] == []
        assert state.integration_report is not None

    @pytest.mark.asyncio
    async def test_the_report_text_names_the_target_and_counts(
        self, integrated_git, integration_repo, settings
    ):
        branch = await agent_branch(
            integrated_git, integration_repo, "reported", "feature.py", "Z = 3\n"
        )
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
        ).integrate(state_with(branch))
        assert "main" in report.report
        assert "1" in report.report

    @pytest.mark.asyncio
    async def test_events_are_emitted_throughout(self, integrated_git, integration_repo, settings):
        from app.events import EventType

        bus = EventBus()
        branch = await agent_branch(
            integrated_git, integration_repo, "eventful", "feature.py", "W = 4\n"
        )
        state = state_with(branch)
        orch_id = state.id
        await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
            event_bus=bus,
            orchestration_id=orch_id,
        ).integrate(state)
        seen = {e.type for e in bus.recent(orch_id)}
        assert EventType.INTEGRATION_STARTED in seen
        assert EventType.INTEGRATION_FINISHED in seen
        assert EventType.TESTS_STARTED in seen
        assert EventType.TESTS_FINISHED in seen

    @pytest.mark.asyncio
    async def test_a_critical_review_finding_downgrades_the_run(
        self, integrated_git, integration_repo, settings
    ):
        branch = await agent_branch(
            integrated_git, integration_repo, "reviewed", "feature.py", "Q = 5\n"
        )
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
            llm=ScriptedLLM(
                responses=[
                    '{"findings": [{"severity": "critical", "file": "feature.py", '
                    '"line": 1, "issue": "hardcoded secret", "fix": "use env"}]}'
                ],
                default='{"findings": [{"severity": "critical", "file": "feature.py", '
                '"line": 1, "issue": "hardcoded secret", "fix": "use env"}]}',
            ),
        ).integrate(state_with(branch))
        assert report.review_findings
        assert any("Critical review finding" in r for r in report.regressions)
        assert report.success is False

    @pytest.mark.asyncio
    async def test_a_minor_review_finding_does_not_downgrade_the_run(
        self, integrated_git, integration_repo, settings
    ):
        branch = await agent_branch(
            integrated_git, integration_repo, "nitpick", "feature.py", "Q = 6\n"
        )
        report = await Integrator(
            integrated_git,
            settings=Settings(_env_file=None),
            test_runner=TestRunner(timeout=120),
            llm=ScriptedLLM(
                responses=[
                    '{"findings": [{"severity": "minor", "file": "feature.py", '
                    '"line": 1, "issue": "naming", "fix": "rename"}]}'
                ],
                default='{"findings": [{"severity": "minor", "file": "feature.py", '
                '"line": 1, "issue": "naming", "fix": "rename"}]}',
            ),
        ).integrate(state_with(branch))
        assert report.review_findings
        assert report.regressions == []
        assert report.success is True

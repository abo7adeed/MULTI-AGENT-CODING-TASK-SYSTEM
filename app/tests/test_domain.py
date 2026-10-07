"""Domain model behaviour: validation, coercion, derived properties."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from app.models.domain import (
    AgentResult,
    DAG,
    ExecutionState,
    FileChange,
    FinalStatus,
    IntegrationReport,
    Project,
    Task,
    TaskStatus,
    TaskType,
    TestRunResult,
)


class TestTask:
    def test_defaults(self):
        task = Task(title="Build API", description="d", type="backend")
        assert task.status is TaskStatus.PENDING
        assert task.priority == 5
        assert task.dependencies == []
        assert task.id and len(task.id) == 36
        assert task.created_at <= time.time()

    def test_terminal_and_retry_predicates(self):
        task = Task(title="t", type="backend")
        assert not task.is_terminal
        assert task.can_retry
        task.status = TaskStatus.SUCCESS
        assert task.is_terminal
        task.status = TaskStatus.FAILED
        task.retry_count = task.max_retries
        assert not task.can_retry

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("backend", TaskType.BACKEND),
            ("BACKEND", TaskType.BACKEND),
            ("ai-ml", TaskType.AI_ML),
            ("ai_ml", TaskType.AI_ML),
            ("ml", TaskType.AI_ML),
            ("rag", TaskType.AI_ML),
            ("db", TaskType.DATABASE),
            ("ui", TaskType.FRONTEND),
            ("dev ops", TaskType.DEVOPS),
            ("tests", TaskType.TESTING),
            ("plan", TaskType.PLANNING),
            ("something-unknown", TaskType.GENERIC),
        ],
    )
    def test_type_coercion_never_crashes_on_freeform_llm_output(self, raw, expected):
        assert Task(title="t", type=raw).type is expected

    def test_self_dependency_rejected(self):
        task = Task(id="fixed", title="t", type="backend")
        object.__setattr__(task, "dependencies", ["fixed"])
        # Re-validate to trigger the field validator.
        with pytest.raises(ValidationError):
            Task(id="fixed", title="t", type="backend", dependencies=["fixed"])


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(task_id="t1", status=TaskStatus.SUCCESS, summary="done")
        assert result.files_changed == []
        assert result.tests_passed == 0
        assert result.commit is None
        assert result.errors == []

    def test_full_payload(self):
        result = AgentResult(
            task_id="t1",
            status=TaskStatus.SUCCESS,
            summary="ok",
            files_changed=["a.py"],
            file_changes=[FileChange(path="a.py", action="created")],
            tests_passed=3,
            tests_failed=0,
            commit="abc123",
        )
        assert result.file_changes[0].action == "created"
        assert result.tests_passed == 3


class TestDAG:
    def test_rebuild_edges_is_derived_from_dependencies(self):
        a = Task(id="a", title="a", type="backend")
        b = Task(id="b", title="b", type="backend", dependencies=["a"])
        c = Task(id="c", title="c", type="backend", dependencies=["a", "b"])
        dag = DAG(tasks={t.id: t for t in (a, b, c)})
        dag.rebuild_edges()
        assert dag.edges["a"] == ["b", "c"]
        assert dag.edges["b"] == ["c"]
        assert dag.dependents_of("a") == ["b", "c"]
        assert dag.size == 3

    def test_add_rebuilds(self):
        dag = DAG()
        dag.add(Task(id="x", title="x", type="backend"))
        dag.add(Task(id="y", title="y", type="backend", dependencies=["x"]))
        assert dag.edges["x"] == ["y"]


class TestProject:
    def test_path_is_normalised(self, tmp_path):
        raw = str(tmp_path / "sub" / ".." / "demo")
        project = Project(name="p", description="d", local_path=raw)
        assert project.local_path == str((tmp_path / "demo").resolve())
        assert project.base_branch == "main"


class TestExecutionState:
    def test_record_error_deduplicates(self):
        state = ExecutionState(project_id="p", original_task="t", repository="/r")
        state.record_error("boom")
        state.record_error("boom")
        assert state.errors == ["boom"]

    def test_defaults(self):
        state = ExecutionState(project_id="p", original_task="t", repository="/r")
        assert state.current_phase.value == "INITIALIZATION"
        assert state.final_status is FinalStatus.PENDING
        assert state.dag.tasks == {}


class TestReports:
    def test_test_run_summary_truncates(self):
        run = TestRunResult(command="pytest", passed=False, stdout="a\nb\nc\n" * 500)
        assert len(run.summary) <= 500

    def test_integration_report_defaults(self):
        report = IntegrationReport()
        assert report.success is False
        assert report.merged_branches == []
        assert report.skipped_details == []

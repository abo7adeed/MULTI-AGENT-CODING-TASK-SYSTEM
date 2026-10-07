"""DAG engine: validation, cycle detection, ready sets, waves, critical path."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.engine.dag import (
    CycleDetectedError,
    DAGEngine,
    DAGError,
    MissingDependencyError,
)
from app.models.domain import DAG, Task, TaskStatus


def task(tid: str, deps=(), priority: int = 5, title: str | None = None) -> Task:
    return Task(
        id=tid,
        title=title or f"T-{tid}",
        type="backend",
        dependencies=list(deps),
        priority=priority,
    )


def build(*tasks: Task) -> DAG:
    return DAG(tasks={t.id: t for t in tasks})


@pytest.fixture
def diamond() -> DAGEngine:
    """
        A
       / \\
      B   C
       \\ /
        D
    """
    return DAGEngine(
        build(
            task("a"),
            task("b", ["a"]),
            task("c", ["a"]),
            task("d", ["b", "c"]),
        )
    )


class TestValidation:
    def test_valid_graph(self, diamond):
        assert diamond.validate() is True

    def test_missing_dependency(self):
        engine = DAGEngine(build(task("a", ["ghost"])))
        with pytest.raises(MissingDependencyError) as exc:
            engine.validate()
        assert "ghost" in str(exc.value)

    def test_self_dependency_rejected_by_the_model(self):
        # The field validator is the real guard: a self-dependency cannot
        # even be constructed, so the DAG can never contain one.
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            task("a", ["a"])

    def test_self_dependency_caught_by_the_engine_too(self):
        # Defence in depth for graphs built by mutating a validated model.
        node = task("a")
        object.__setattr__(node, "dependencies", ["a"])
        with pytest.raises(DAGError, match="cannot depend on itself"):
            DAGEngine(build(node)).validate()

    @pytest.mark.parametrize(
        "cycle",
        [
            ["a", "b"],
            ["a", "b", "c"],
        ],
    )
    def test_cycle_detection_reports_path(self, cycle):
        tasks = [task(node, [cycle[(i + 1) % len(cycle)]]) for i, node in enumerate(cycle)]
        engine = DAGEngine(build(*tasks))
        with pytest.raises(CycleDetectedError) as exc:
            engine.validate()
        assert exc.value.path[0] == exc.value.path[-1]
        assert len(exc.value.path) == len(cycle) + 1

    def test_dag_error_is_value_error(self):
        assert issubclass(DAGError, ValueError)

    def test_unknown_task_lookup(self, diamond):
        with pytest.raises(KeyError, match="Unknown task"):
            diamond.get("nope")


class TestReadyTasks:
    def test_respects_dependencies(self, diamond):
        assert diamond.get_ready_tasks(set()) == ["a"]
        assert set(diamond.get_ready_tasks({"a"})) == {"b", "c"}
        assert diamond.get_ready_tasks({"a", "b", "c"}) == ["d"]
        assert diamond.get_ready_tasks({"a", "b", "c", "d"}) == []

    def test_does_not_re_return_completed(self, diamond):
        assert "a" not in diamond.get_ready_tasks({"a"})

    def test_priority_ordering(self):
        engine = DAGEngine(
            build(task("low", priority=9), task("high", priority=1), task("mid", priority=5))
        )
        assert engine.get_ready_tasks(set()) == ["high", "mid", "low"]


class TestWaves:
    def test_parallel_waves(self, diamond):
        assert diamond.parallel_waves() == [["a"], ["b", "c"], ["d"]]

    def test_independent_tasks_share_a_wave(self):
        engine = DAGEngine(build(*[task(f"t{i}") for i in range(5)]))
        assert engine.parallel_waves() == [[f"t{i}" for i in range(5)]]

    def test_max_parallelism(self, diamond):
        assert diamond.estimated_parallelism() == 2
        assert diamond.estimated_parallelism(1) == 1
        assert diamond.estimated_parallelism(8) == 2

    def test_critical_path(self, diamond):
        # b and c are at the same depth; the path is a valid longest chain.
        assert diamond.critical_path()[0] == "a"
        assert diamond.critical_path()[-1] == "d"
        assert len(diamond.critical_path()) == 3

    def test_critical_path_prefers_the_load_bearing_branch(self):
        # `a -> b -> e` is strictly longer than `a -> c`, so it wins outright.
        engine = DAGEngine(
            build(
                task("a"),
                task("b", ["a"]),
                task("c", ["a"]),
                task("e", ["b"]),
            )
        )
        assert engine.critical_path() == ["a", "b", "e"]

    def test_critical_path_breaks_ties_by_descendant_count(self):
        # b and c are the same depth; b has the larger downstream subtree.
        engine = DAGEngine(
            build(
                task("a"),
                task("b", ["a"]),
                task("c", ["a"]),
                task("tail", ["b"]),
            )
        )
        assert engine.critical_path() == ["a", "b", "tail"]

    def test_empty_graph(self):
        engine = DAGEngine(DAG())
        assert engine.parallel_waves() == []
        assert engine.critical_path() == []
        assert engine.summary()["total_tasks"] == 0


class TestReachability:
    def test_ancestors(self, diamond):
        assert diamond.ancestors("d") == {"a", "b", "c"}

    def test_descendants(self, diamond):
        assert diamond.descendants("a") == {"b", "c", "d"}

    def test_roots_and_leaves(self, diamond):
        assert diamond.roots() == ["a"]
        assert set(diamond.leaves()) == {"d"}


class TestTopologicalOrder:
    def test_order_respects_dependencies(self, diamond):
        order = diamond.topological_order()
        assert order[0] == "a"
        assert order[-1] == "d"
        positions = {t: i for i, t in enumerate(order)}
        for node in diamond.tasks.values():
            for dep in node.dependencies:
                assert positions[dep] < positions[node.id]

    def test_raises_on_cycle(self):
        engine = DAGEngine(build(task("a", ["b"]), task("b", ["a"])))
        with pytest.raises(CycleDetectedError):
            engine.topological_order()


class TestBlocked:
    def test_transitive_propagation(self, diamond):
        blocked = dict(diamond.get_blocked_tasks({"a"}))
        assert set(blocked) == {"b", "c", "d"}
        assert "T-a" in blocked["b"]

    def test_upstream_of_a_failure_is_not_blocked(self, diamond):
        # Only *dependents* of a failure are blocked. A cancelled leaf leaves
        # everything that ran before it intact.
        assert diamond.get_blocked_tasks({"d"}) == []

    def test_only_dependents_are_reported(self, diamond):
        # d depends on b and c, so b's failure blocks d. c is untouched.
        assert {t for t, _ in diamond.get_blocked_tasks({"b"})} == {"d"}

    def test_already_failed_tasks_are_not_re_reported(self, diamond):
        # A task already in the failure set is not also "blocked"; the caller
        # already knows about it.
        assert {t for t, _ in diamond.get_blocked_tasks({"b", "d"})} == set()

    def test_nothing_blocked_when_upstream_succeeds(self, diamond):
        assert diamond.get_blocked_tasks(set()) == []


class TestSyncStatuses:
    def test_marks_blocked_and_ready(self, diamond):
        diamond.tasks["a"].status = TaskStatus.FAILED
        diamond.sync_statuses()
        assert diamond.tasks["b"].status is TaskStatus.BLOCKED
        assert diamond.tasks["b"].blocked_reason

    def test_marks_ready_when_dependencies_succeed(self, diamond):
        diamond.tasks["a"].status = TaskStatus.SUCCESS
        diamond.sync_statuses()
        assert diamond.tasks["b"].status is TaskStatus.READY
        assert diamond.tasks["d"].status is TaskStatus.PENDING


class TestSummary:
    def test_shape(self, diamond):
        summary = diamond.summary()
        assert summary["total_tasks"] == 4
        assert summary["max_wave_size"] == 2
        assert summary["edges"]["a"] == ["b", "c"]
        assert "PENDING" in summary["status_counts"]

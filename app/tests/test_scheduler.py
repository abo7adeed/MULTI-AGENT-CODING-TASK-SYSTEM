"""
Scheduler behaviour.

These are the guarantees the whole system rests on: bounded concurrency, real
retries, failure propagation, and a cancel that actually stops work.
"""

from __future__ import annotations

import asyncio

import pytest

from app.engine.dag import CycleDetectedError, DAGEngine
from app.engine.scheduler import RetryPolicy, Scheduler
from app.models.domain import DAG, Task, TaskStatus

pytestmark = pytest.mark.asyncio


def task(tid: str, deps=(), priority: int = 5) -> Task:
    return Task(
        id=tid, title=f"T-{tid}", type="backend", dependencies=list(deps), priority=priority
    )


def dag_of(*tasks: Task) -> DAG:
    return DAG(tasks={t.id: t for t in tasks})


FAST = RetryPolicy(max_retries=2, base_delay=0.01, multiplier=1.0)
NONE = RetryPolicy(max_retries=0, base_delay=0.0, multiplier=1.0)


class Tracker:
    """Records execution order and peak concurrency."""

    def __init__(self, fail: set[str] | None = None, delay: float = 0.02) -> None:
        self.fail = fail or set()
        self.delay = delay
        self.order: list[str] = []
        self.live = 0
        self.peak = 0
        self.attempts: dict[str, int] = {}

    async def __call__(self, t: Task) -> None:
        self.live += 1
        self.peak = max(self.peak, self.live)
        self.order.append(t.id)
        self.attempts[t.id] = self.attempts.get(t.id, 0) + 1
        try:
            await asyncio.sleep(self.delay)
            if t.id in self.fail:
                raise RuntimeError(f"{t.id} exploded")
        finally:
            self.live -= 1


class TestHappyPath:
    async def test_all_tasks_run_in_order(self):
        graph = dag_of(task("a"), task("b", ["a"]), task("c", ["a"]))
        tracker = Tracker()
        stats = await Scheduler(DAGEngine(graph), tracker, max_concurrency=4).run()

        assert stats.succeeded == 3
        assert stats.failed == 0
        assert stats.error is None
        assert tracker.order[0] == "a"
        assert set(tracker.order[1:]) == {"b", "c"}

    async def test_timings_recorded(self):
        graph = dag_of(task("a"))
        await Scheduler(DAGEngine(graph), Tracker(), max_concurrency=2).run()
        t = graph.tasks["a"]
        assert t.started_at is not None
        assert t.finished_at is not None
        assert t.duration_seconds >= 0


class TestConcurrency:
    async def test_cap_is_respected(self):
        graph = dag_of(*[task(f"t{i}") for i in range(12)])
        stats = await Scheduler(DAGEngine(graph), Tracker(delay=0.02), max_concurrency=3).run()
        assert stats.peak_concurrency <= 3
        assert stats.succeeded == 12

    async def test_cap_of_one_serialises(self):
        graph = dag_of(*[task(f"t{i}") for i in range(4)])
        tracker = Tracker(delay=0.01)
        await Scheduler(DAGEngine(graph), tracker, max_concurrency=1).run()
        assert tracker.peak == 1

    async def test_independent_tasks_actually_overlap(self):
        graph = dag_of(*[task(f"t{i}") for i in range(4)])
        tracker = Tracker(delay=0.05)
        await Scheduler(DAGEngine(graph), tracker, max_concurrency=4).run()
        assert tracker.peak > 1


class TestRetries:
    async def test_transient_failure_is_retried_then_succeeds(self):
        calls: dict[str, int] = {}

        async def flaky(t: Task) -> None:
            calls[t.id] = calls.get(t.id, 0) + 1
            if calls[t.id] < 3:
                raise RuntimeError("transient")

        graph = dag_of(task("a"))
        stats = await Scheduler(
            DAGEngine(graph), flaky, max_concurrency=2, retry_policy=FAST
        ).run()
        assert stats.succeeded == 1
        assert stats.retries == 2
        assert graph.tasks["a"].retry_count == 2
        assert len(graph.tasks["a"].errors) == 2

    async def test_retries_are_bounded(self):
        async def always_fail(t: Task) -> None:
            raise RuntimeError("nope")

        graph = dag_of(task("a"))
        policy = RetryPolicy(max_retries=3, base_delay=0.0)
        stats = await Scheduler(
            DAGEngine(graph), always_fail, max_concurrency=2, retry_policy=policy
        ).run()
        assert stats.failed == 1
        assert calls_of(graph.tasks["a"]) == 4  # 1 initial + 3 retries
        assert graph.tasks["a"].status is TaskStatus.FAILED

    async def test_backoff_grows(self):
        policy = RetryPolicy(max_retries=4, base_delay=1.0, multiplier=2.0, max_delay=10.0)
        assert [policy.delay_for(i) for i in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]
        assert policy.delay_for(10) == 10.0  # capped

    async def test_zero_retries_fails_immediately(self):
        async def always_fail(t: Task) -> None:
            raise RuntimeError("nope")

        graph = dag_of(task("a"))
        await Scheduler(
            DAGEngine(graph), always_fail, max_concurrency=1, retry_policy=NONE
        ).run()
        assert calls_of(graph.tasks["a"]) == 1


class TestFailurePropagation:
    async def test_downstream_marked_blocked_with_reason(self):
        graph = dag_of(task("a"), task("b", ["a"]), task("c", ["b"]))
        stats = await Scheduler(
            DAGEngine(graph), Tracker(fail={"a"}), max_concurrency=4, retry_policy=NONE
        ).run()
        assert stats.failed == 1
        assert stats.blocked == 2
        assert graph.tasks["a"].status is TaskStatus.FAILED
        assert graph.tasks["b"].status is TaskStatus.BLOCKED
        assert "T-a" in graph.tasks["b"].blocked_reason
        assert graph.tasks["c"].status is TaskStatus.BLOCKED

    async def test_siblings_still_run(self):
        graph = dag_of(task("a"), task("b", ["a"]), task("independent"))
        await Scheduler(
            DAGEngine(graph), Tracker(fail={"a"}), max_concurrency=4, retry_policy=NONE
        ).run()
        assert graph.tasks["independent"].status is TaskStatus.SUCCESS

    async def test_no_task_left_non_terminal(self):
        graph = dag_of(task("a"), task("b", ["a"]), task("c", ["b"]))
        await Scheduler(
            DAGEngine(graph), Tracker(fail={"a"}), max_concurrency=2, retry_policy=NONE
        ).run()
        assert all(t.status.is_terminal for t in graph.tasks.values())
        assert all(t.is_terminal for t in graph.tasks.values())


class TestTimeout:
    async def test_slow_task_times_out(self):
        async def slow(t: Task) -> None:
            await asyncio.sleep(5)

        graph = dag_of(task("a"))
        stats = await Scheduler(
            DAGEngine(graph), slow, task_timeout=0.05, max_concurrency=1, retry_policy=NONE
        ).run()
        assert stats.failed == 1
        assert "timeout" in graph.tasks["a"].errors[0].lower()


class TestCancel:
    async def test_cancel_stops_in_flight_work(self):
        async def slow(t: Task) -> None:
            await asyncio.sleep(5)

        graph = dag_of(*[task(f"t{i}") for i in range(6)])
        scheduler = Scheduler(DAGEngine(graph), slow, max_concurrency=3)
        job = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        scheduler.cancel()
        stats = await job

        assert stats.cancelled == 6
        assert stats.succeeded == 0
        assert all(t.status is TaskStatus.CANCELLED for t in graph.tasks.values())

    async def test_cancel_prevents_further_dispatch(self):
        started: list[str] = []

        async def slow(t: Task) -> None:
            started.append(t.id)
            await asyncio.sleep(0.3)

        graph = dag_of(*[task(f"t{i}") for i in range(10)])
        scheduler = Scheduler(DAGEngine(graph), slow, max_concurrency=2)
        job = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        scheduler.cancel()
        await job
        assert len(started) <= 2


class TestPauseResume:
    async def test_pause_halts_dispatch_until_resumed(self):
        graph = dag_of(*[task(f"t{i}") for i in range(5)])
        tracker = Tracker(delay=0.02)
        scheduler = Scheduler(DAGEngine(graph), tracker, max_concurrency=1)
        job = asyncio.create_task(scheduler.run())

        await asyncio.sleep(0.03)
        scheduler.pause()
        await asyncio.sleep(0.12)
        mid = len(tracker.order)
        scheduler.resume()
        stats = await job

        assert mid < 5
        assert stats.succeeded == 5
        assert scheduler.is_paused is False

    async def test_cancel_unblocks_a_paused_scheduler(self):
        graph = dag_of(*[task(f"t{i}") for i in range(10)])

        async def slow(t: Task) -> None:
            await asyncio.sleep(5)

        scheduler = Scheduler(DAGEngine(graph), slow, max_concurrency=1)
        job = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        scheduler.pause()
        await asyncio.sleep(0.05)
        scheduler.cancel()
        stats = await asyncio.wait_for(job, timeout=5)
        assert stats.cancelled > 0


class TestInvalidGraph:
    async def test_cycle_reported_not_crashed(self):
        graph = dag_of(task("a", ["b"]), task("b", ["a"]))

        async def never(t: Task) -> None:  # pragma: no cover - must not run
            raise AssertionError("a task ran in an invalid graph")

        stats = await Scheduler(DAGEngine(graph), never).run()
        assert "Cycle" in (stats.error or "")
        assert stats.dispatched == 0

    async def test_missing_dependency_reported(self):
        graph = dag_of(task("a", ["ghost"]))
        stats = await Scheduler(DAGEngine(graph), Tracker()).run()
        assert "ghost" in (stats.error or "")


def calls_of(t: Task) -> int:
    return t.retry_count + 1

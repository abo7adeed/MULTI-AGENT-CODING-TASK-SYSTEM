"""
Scheduler -- deterministic async execution policy.

Owns everything the pure `DAGEngine` deliberately does not:

  * priority-ordered dispatch
  * a hard concurrency cap (no unbounded fan-out)
  * bounded exponential-backoff retries, per task
  * failure propagation: tasks whose dependency failed become BLOCKED
  * cooperative pause / resume / cancel
  * per-task timeouts
  * event emission for the live UI

The runner is injected, so the scheduler is testable with a trivial lambda and
unchanged by swapping MockAgent for a real LLM agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from app.engine.dag import DAGEngine, DAGError
from app.events import EventBus, EventType, get_event_bus
from app.logging_config import get_logger, reset_task_scope, task_scope
from app.models.domain import Task, TaskStatus

logger = get_logger("app.engine.scheduler")

TaskRunner = Callable[[Task], Awaitable[None]]


class Cancelled(Exception):
    """Raised inside a task runner when the run is cancelled."""


@dataclass
class RetryPolicy:
    """Bounded exponential backoff. Guarantees termination."""

    max_retries: int = 3
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 30.0

    def delay_for(self, attempt: int) -> float:
        """`attempt` is 1-based: the delay to wait *after* that failure."""
        return min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)

    def should_retry(self, attempt: int) -> bool:
        return attempt <= self.max_retries


@dataclass
class SchedulerStats:
    dispatched: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    blocked: int = 0
    cancelled: int = 0
    peak_concurrency: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


class Scheduler:
    """
    Executes a DAG.

        scheduler = Scheduler(engine, runner, orchestration_id="abc")
        stats = await scheduler.run()
    """

    def __init__(
        self,
        engine: DAGEngine,
        runner: TaskRunner,
        orchestration_id: str = "",
        max_concurrency: int = 4,
        retry_policy: Optional[RetryPolicy] = None,
        event_bus: Optional[EventBus] = None,
        task_timeout: float = 1800.0,
        on_state_change: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.engine = engine
        self.runner = runner
        self.orchestration_id = orchestration_id
        self.max_concurrency = max(1, max_concurrency)
        self.retry_policy = retry_policy or RetryPolicy()
        self.event_bus = event_bus or get_event_bus()
        self.task_timeout = task_timeout
        self._on_state_change = on_state_change

        self.succeeded: set[str] = set()
        self.failed: set[str] = set()
        self.blocked: set[str] = set()
        self.cancelled_tasks: set[str] = set()
        self.running: set[str] = set()
        self.stats = SchedulerStats()

        self._in_flight: dict[asyncio.Task, str] = {}
        self._pause_event = asyncio.Event()
        self._pause_event.set()          # set == running
        self._cancel_event = asyncio.Event()
        self._pending_blocked_events: list[str] = []
        self._notified_count = 0

    # ── public control surface ──────────────────────────────────────────────

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def cancel(self) -> None:
        """
        Stop the run now.

        Aborts in-flight tasks immediately rather than letting them drain: a
        user pressing Cancel expects the agents to actually stop, and an agent
        mid-LLM-call would otherwise keep burning tokens until it finished.
        """
        self._cancel_event.set()
        self._pause_event.set()  # unblock a paused scheduler so it can exit
        for handle in list(self._in_flight):
            handle.cancel()

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self) -> SchedulerStats:
        started = time.monotonic()
        try:
            self.engine.dag.rebuild_edges()
            self.engine.validate()
        except DAGError as exc:
            self.stats.error = str(exc)
            self.stats.duration_seconds = round(time.monotonic() - started, 3)
            logger.error("Invalid DAG", extra={"error": str(exc)})
            await self._emit(EventType.ORCHESTRATION_FINISHED, error=str(exc))
            return self.stats

        total = len(self.engine.tasks)
        self.stats.extra["total"] = total
        await self._emit(
            EventType.ORCHESTRATION_STARTED,
            total_tasks=total,
            max_concurrency=self.max_concurrency,
            waves=len(self.engine.parallel_waves()),
        )

        try:
            await self._loop(total)
        except asyncio.CancelledError:
            self.cancel()
            for task_id in list(self._in_flight.values()):
                self.engine.tasks[task_id].status = TaskStatus.CANCELLED
            await self._cancel_in_flight()
            raise
        except Exception as exc:  # noqa: BLE001 - a run must never die silently
            self.stats.error = str(exc)
            logger.exception("Scheduler loop crashed", extra={"error": str(exc)})
            await self._cancel_in_flight()
        finally:
            # Never leave an orphaned task running behind a finished scheduler.
            if self._in_flight:
                await self._abandon_in_flight()
            self.stats.duration_seconds = round(time.monotonic() - started, 3)

        self._settle_remaining(total)
        await self._emit(
            EventType.ORCHESTRATION_FINISHED,
            succeeded=len(self.succeeded),
            failed=len(self.failed),
            blocked=len(self.blocked),
            cancelled=len(self.cancelled_tasks),
            retries=self.stats.retries,
            duration_seconds=self.stats.duration_seconds,
        )
        return self.stats

    async def _loop(self, total: int) -> None:
        while True:
            if self.is_cancelled:
                break

            if self.is_paused:
                await self._emit(EventType.ORCHESTRATION_PAUSED)
                await self._pause_event.wait()
                if self.is_cancelled:
                    break
                await self._emit(EventType.ORCHESTRATION_RESUMED)

            await self._reap()

            if self._settled(total):
                break

            # Dispatch everything ready, up to the concurrency cap.
            slots = self.max_concurrency - len(self._in_flight)
            dispatched = 0
            if slots > 0:
                settled = self._settled_ids()
                ready = [
                    tid
                    for tid in self.engine.get_ready_tasks(self.succeeded)
                    if tid not in settled and tid not in self.running
                ][:slots]
                for task_id in ready:
                    self._dispatch(task_id)
                    dispatched += 1

            if self._in_flight:
                await self._wait_for_one()
                continue

            # Nothing running. If we just dispatched, let the event loop
            # register the handles before drawing conclusions.
            if dispatched:
                await asyncio.sleep(0)
                continue

            # Nothing running and nothing was runnable => the rest can never
            # execute. Record why, rather than leaving tasks PENDING forever.
            if self._settled(total):
                break
            stranded = [tid for tid in self.engine.tasks if tid not in self._settled_ids()]
            if not stranded:
                break
            self._mark_blocked(stranded)
            break

    # ── dispatch & completion ───────────────────────────────────────────────

    def _dispatch(self, task_id: str) -> None:
        task = self.engine.tasks[task_id]
        task.status = TaskStatus.RUNNING
        self.running.add(task_id)
        self.stats.dispatched += 1
        self.stats.peak_concurrency = max(self.stats.peak_concurrency, len(self.running))
        handle = asyncio.create_task(self._run_task(task), name=f"task:{task_id[:8]}")
        self._in_flight[handle] = task_id

    async def _run_task(self, task: Task) -> None:
        token = task_scope(task.id)
        attempt = 0
        try:
            while True:
                attempt += 1
                if self.is_cancelled:
                    self._mark_cancelled(task)
                    return

                task.retry_count = max(task.retry_count, attempt - 1)
                if task.started_at is None:
                    task.started_at = time.time()
                await self._emit(
                    EventType.TASK_STARTED,
                    task_id=task.id,
                    title=task.title,
                    type=task.type.value,
                    agent=task.assigned_agent,
                    attempt=attempt,
                )
                await self._notify()

                try:
                    await asyncio.wait_for(self.runner(task), timeout=self.task_timeout)
                    self._mark_success(task)
                    return
                except Cancelled:
                    self._mark_cancelled(task)
                    return
                except asyncio.TimeoutError:
                    error: Exception = TimeoutError(
                        f"Task exceeded {self.task_timeout}s timeout"
                    )
                except Exception as exc:  # noqa: BLE001 - recorded, then retried
                    error = exc

                task.errors.append(f"attempt {attempt}: {error}")
                task.output["last_error"] = str(error)
                logger.warning(
                    "Task attempt failed",
                    extra={
                        "task_title": task.title,
                        "attempt": attempt,
                        "error": str(error)[:400],
                    },
                )

                if not self.retry_policy.should_retry(attempt):
                    self._mark_failed(task, str(error))
                    return

                delay = self.retry_policy.delay_for(attempt)
                task.status = TaskStatus.RETRYING
                self.stats.retries += 1
                await self._emit(
                    EventType.TASK_RETRYING,
                    task_id=task.id,
                    title=task.title,
                    attempt=attempt,
                    retry_in=round(delay, 2),
                    error=str(error)[:300],
                )
                await self._notify()
                await asyncio.sleep(delay)

                if self.is_cancelled:
                    self._mark_cancelled(task)
                    return
        finally:
            self.running.discard(task.id)
            with contextlib.suppress(Exception):
                reset_task_scope(token)

    def _mark_success(self, task: Task) -> None:
        task.status = TaskStatus.SUCCESS
        task.finished_at = time.time()
        if task.started_at:
            task.duration_seconds = round(task.finished_at - task.started_at, 3)
        self.succeeded.add(task.id)
        self.stats.succeeded += 1

    def _mark_failed(self, task: Task, reason: str) -> None:
        task.status = TaskStatus.FAILED
        task.finished_at = time.time()
        if task.started_at:
            task.duration_seconds = round(task.finished_at - task.started_at, 3)
        self.failed.add(task.id)
        self.stats.failed += 1
        logger.error(
            "Task failed permanently",
            extra={"task_title": task.title, "reason": reason[:400]},
        )

    def _mark_cancelled(self, task: Task) -> None:
        task.status = TaskStatus.CANCELLED
        task.finished_at = time.time()
        self.cancelled_tasks.add(task.id)
        self.stats.cancelled += 1

    def _mark_blocked(self, task_ids: list[str]) -> None:
        """Record tasks that can never run, with a human-readable reason."""
        reasons = dict(self.engine.get_blocked_tasks(self.failed | self.cancelled_tasks))
        for task_id in task_ids:
            task = self.engine.tasks.get(task_id)
            if task is None or task_id in self._settled_ids():
                continue
            task.status = TaskStatus.BLOCKED
            task.blocked_reason = reasons.get(
                task_id, "no runnable dependency path (upstream failure or cancellation)"
            )
            self.blocked.add(task_id)
            self.stats.blocked += 1
        self._pending_blocked_events.extend(task_ids)

    # ── loop plumbing ───────────────────────────────────────────────────────

    async def _reap(self) -> None:
        """Non-blocking harvest of already-finished in-flight tasks."""
        if not self._in_flight:
            return
        done, _ = await asyncio.wait(
            set(self._in_flight.keys()), timeout=0, return_when=asyncio.ALL_COMPLETED
        )
        if done:
            await self._harvest(done)

    async def _wait_for_one(self) -> None:
        """Block until at least one in-flight task completes."""
        if not self._in_flight:
            await asyncio.sleep(0.01)
            return
        done, _pending = await asyncio.wait(
            set(self._in_flight.keys()), return_when=asyncio.FIRST_COMPLETED
        )
        await self._harvest(done)

    async def _harvest(self, done: set[asyncio.Task]) -> None:
        """
        The single place a finished task is retired and its terminal event is
        emitted. Shared by the non-blocking reap and the blocking wait so the
        two paths cannot drift apart.
        """
        for handle in done:
            task_id = self._in_flight.pop(handle, None)
            if task_id is None:
                continue
            task = self.engine.tasks[task_id]
            try:
                handle.result()
            except asyncio.CancelledError:
                self._mark_cancelled(task)
            except Exception as exc:  # runner blew up outside our guard
                self._mark_failed(task, str(exc))
            if handle.cancelled() and task_id not in self._settled_ids():
                self._mark_cancelled(task)
            await self._emit_task_terminal(task)

        if self._pending_blocked_events:
            for task_id in self._pending_blocked_events:
                task = self.engine.tasks.get(task_id)
                if task is None:
                    continue
                await self._emit(
                    EventType.TASK_BLOCKED,
                    task_id=task_id,
                    title=task.title,
                    reason=task.blocked_reason,
                )
            self._pending_blocked_events = []
        await self._notify()

    async def _emit_task_terminal(self, task: Task) -> None:
        if task.status == TaskStatus.SUCCESS:
            await self._emit(
                EventType.TASK_SUCCEEDED,
                task_id=task.id,
                title=task.title,
                agent=task.assigned_agent,
                type=task.type.value,
                duration_seconds=task.duration_seconds,
                retries=task.retry_count,
                files_changed=task.output.get("files_changed", []),
            )
        elif task.status == TaskStatus.FAILED:
            await self._emit(
                EventType.TASK_FAILED,
                task_id=task.id,
                title=task.title,
                agent=task.assigned_agent,
                type=task.type.value,
                errors=task.errors[-3:],
                duration_seconds=task.duration_seconds,
                retries=task.retry_count,
            )
        elif task.status == TaskStatus.CANCELLED:
            await self._emit(EventType.TASK_CANCELLED, task_id=task.id, title=task.title)

    async def _cancel_in_flight(self) -> None:
        for handle in list(self._in_flight):
            handle.cancel()
        if self._in_flight:
            await asyncio.gather(*list(self._in_flight), return_exceptions=True)
        self._in_flight.clear()

    async def _abandon_in_flight(self) -> None:
        """Stop and retire tasks still running when the loop exits."""
        for handle, task_id in list(self._in_flight.items()):
            handle.cancel()
        if self._in_flight:
            await asyncio.gather(*list(self._in_flight), return_exceptions=True)
        for task_id in list(self._in_flight.values()):
            self._in_flight.pop(task_id, None)
        self._in_flight.clear()
        self.running.clear()

    def _settled_ids(self) -> set[str]:
        return self.succeeded | self.failed | self.cancelled_tasks | self.blocked

    def _settled(self, total: int) -> bool:
        return len(self._settled_ids()) >= total

    def _settle_remaining(self, total: int) -> None:
        """After the loop, no task may be left in a non-terminal state."""
        settled = self._settled_ids()
        remaining = [tid for tid in self.engine.tasks if tid not in settled]
        if not remaining:
            self._pending_blocked_events = []
            self.engine.sync_statuses()
            return
        # A run that was cancelled leaves unstarted work cancelled, not blocked.
        if self.is_cancelled:
            for task_id in remaining:
                task = self.engine.tasks[task_id]
                if task.status not in (TaskStatus.RUNNING,):
                    self._mark_cancelled(task)
        else:
            self._mark_blocked(remaining)
            for task_id in self._pending_blocked_events:
                task = self.engine.tasks.get(task_id)
                if task is not None:
                    task.status = TaskStatus.BLOCKED
        self._pending_blocked_events = []
        self.engine.sync_statuses()

    # ── plumbing ────────────────────────────────────────────────────────────

    async def _emit(self, event_type: EventType, **data) -> None:
        if not self.orchestration_id:
            return
        with contextlib.suppress(Exception):
            await self.event_bus.publish(self.orchestration_id, event_type, **data)

    async def _notify(self) -> None:
        """Throttled persistence hook: never more often than every 250ms."""
        if not self._on_state_change:
            return
        now = time.monotonic()
        if now - self._notified_count < 0.25:
            return
        self._notified_count = now
        with contextlib.suppress(Exception):
            await self._on_state_change()

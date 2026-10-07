"""
ExecutionManager -- run lifecycle.

Owns everything *around* a run rather than inside it:

  * starting an orchestration as a background job
  * routing pause / resume / cancel to the live `Scheduler`
  * persisting state on every transition so a crash is recoverable
  * persisting the event stream for replay
  * looking up live schedulers for the API's control endpoints

Keeping this separate from `Scheduler` means the scheduler stays a pure
execution policy with no notion of HTTP, storage or process lifetime.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any, Optional

from app.config import Settings, get_settings
from app.engine.scheduler import RetryPolicy, Scheduler
from app.events import EventBus, EventType, get_event_bus
from app.logging_config import get_logger, orchestration_scope, reset_orchestration_scope
from app.models.domain import (
    ExecutionState,
    FinalStatus,
    OrchestrationStatus,
    Phase,
    TaskStatus,
)
from app.models.orchestration import Orchestration
from app.models.store import InMemoryStateStore, StateStore

logger = get_logger("app.engine.execution")


class RepositoryBusy(RuntimeError):
    """
    Raised when a run would work on a repository another run already holds.

    Isolation is per repository, not per run: agents commit branches into one
    shared `.git`, and the integrator merges, resets and checks out the target
    branch. Two live runs against the same path therefore interleave inside a
    single repository and corrupt each other's merges, so the second one is
    refused rather than started.
    """


class ExecutionManager:
    """
    Single owner of running orchestrations.

    One instance per process. The API holds it; tests construct it directly.
    """

    def __init__(
        self,
        store: Optional[StateStore] = None,
        event_bus: Optional[EventBus] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.store: StateStore = store or InMemoryStateStore()
        self.event_bus = event_bus or get_event_bus()
        self._schedulers: dict[str, Scheduler] = {}
        self._jobs: dict[str, asyncio.Task] = {}
        #: Live runs, keyed by orchestration id, valued by the repository they
        #: hold. Read and written only while `_lock` is held.
        self._repos: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def normalise_repo(path: Optional[str]) -> str:
        """Compare repository paths by identity, not by spelling."""
        if not path:
            return ""
        try:
            return os.path.normcase(os.path.realpath(path))
        except OSError:  # pragma: no cover - defensive
            return os.path.normcase(os.path.abspath(path))

    def _active_on_repo(self, repo: str, exclude: str = "") -> Optional[str]:
        """Id of a live run holding `repo`, if any. Caller holds `_lock`."""
        for other_id, other_repo in self._repos.items():
            if other_id == exclude or other_repo != repo or not repo:
                continue
            job = self._jobs.get(other_id)
            if job is not None and not job.done():
                return other_id
        return None

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def start(
        self,
        orchestration: Orchestration,
        run_fn,
    ) -> Orchestration:
        """
        Launch `orchestration` in the background.

        `run_fn(on_state_change=...) -> ExecutionState` runs the whole
        pipeline. It is supplied by the Orchestrator rather than constructed
        here, because the ordering, retries and integration logic all live on
        that side of the boundary; this method owns lifetime, persistence and
        terminal status only.
        """
        orch_id = orchestration.id
        repo = self.normalise_repo(orchestration.state.repository)
        async with self._lock:
            if orch_id in self._jobs and not self._jobs[orch_id].done():
                raise RuntimeError(f"Orchestration {orch_id} is already running")

            busy = self._active_on_repo(repo, exclude=orch_id)
            if busy is not None:
                raise RepositoryBusy(
                    f"orchestration {busy} is already running on {orchestration.state.repository}; "
                    "agents share one repository's worktrees and branches, so a second "
                    "concurrent run would corrupt both. Wait for it to finish or cancel it."
                )

            orchestration.status = OrchestrationStatus.RUNNING
            orchestration.started_at = orchestration.started_at or time.time()
            orchestration.finished_at = None
            orchestration.state.final_status = FinalStatus.RUNNING
            orchestration.state.started_at = orchestration.state.started_at or time.time()
            orchestration.total_tasks = len(orchestration.state.tasks)
            await self._persist(orchestration)
            await self.event_bus.publish(
                orch_id, EventType.ORCHESTRATION_STARTED, name=orchestration.name
            )

            job = asyncio.create_task(
                self._supervise(orchestration, run_fn), name=f"orch:{orch_id[:8]}"
            )
            self._jobs[orch_id] = job
            self._repos[orch_id] = repo
        return orchestration

    async def _supervise(self, orchestration: Orchestration, run_fn) -> None:
        orch_id = orchestration.id
        token = orchestration_scope(orch_id)
        try:
            await asyncio.wait_for(
                run_fn(on_state_change=lambda: self._persist(orchestration)),
                timeout=self.settings.orchestrator_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._cancel_scheduler(orch_id)
            orchestration.state.record_error(
                f"Orchestration exceeded {self.settings.orchestrator_timeout_seconds}s"
            )
            await self._finalise(orchestration, None, "timeout")
        except asyncio.CancelledError:
            self._cancel_scheduler(orch_id)
            await self._finalise(orchestration, None, "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - a run must never die silently
            logger.exception("Orchestration crashed", extra={"orchestration": orch_id})
            self._cancel_scheduler(orch_id)
            orchestration.state.record_error(f"Orchestration crashed: {exc}")
            await self._finalise(orchestration, None, str(exc))
        else:
            await self._finalise(orchestration, None, None)
        finally:
            self._schedulers.pop(orch_id, None)
            self._repos.pop(orch_id, None)
            with contextlib.suppress(Exception):
                reset_orchestration_scope(token)

    def register_scheduler(self, orch_id: str, scheduler: Scheduler) -> None:
        """Called by the orchestrator so pause/resume/cancel can find the run."""
        self._schedulers[orch_id] = scheduler

    def _cancel_scheduler(self, orch_id: str) -> None:
        scheduler = self._schedulers.get(orch_id)
        if scheduler is not None:
            scheduler.cancel()

    async def _finalise(
        self,
        orchestration: Orchestration,
        scheduler: Optional[Scheduler],
        error: Optional[str],
    ) -> None:
        state = orchestration.state
        state.touch()
        state.finished_at = time.time()

        tasks = list(state.tasks.values())
        succeeded = sum(1 for t in tasks if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        blocked = sum(1 for t in tasks if t.status == TaskStatus.BLOCKED)
        cancelled = sum(1 for t in tasks if t.status == TaskStatus.CANCELLED)
        retried = sum(1 for t in tasks if t.retry_count > 0)
        duration = round(
            (time.time() - (orchestration.started_at or time.time())), 2
        )

        orchestration.completed_tasks = succeeded
        orchestration.failed_tasks = failed
        orchestration.running_tasks = 0

        # A verdict the pipeline already reached. Task counts alone cannot see
        # that every agent "succeeded" while integration merged nothing, and
        # that run must not be reported as a clean success.
        pipeline_failed = state.final_status is FinalStatus.FAILED

        if orchestration.cancel_requested:
            state.final_status = FinalStatus.CANCELLED
            orchestration.status = OrchestrationStatus.CANCELLED
            state.current_phase = Phase.CANCELLED
        elif error:
            state.final_status = FinalStatus.FAILED
            orchestration.status = OrchestrationStatus.FAILED
            state.current_phase = Phase.FAILED
        elif pipeline_failed and succeeded:
            state.final_status = FinalStatus.FAILED
            orchestration.status = OrchestrationStatus.FAILED
            state.current_phase = Phase.FAILED
        elif succeeded == len(tasks) and tasks:
            state.final_status = FinalStatus.SUCCESS
            orchestration.status = OrchestrationStatus.COMPLETED
        elif succeeded:
            state.final_status = FinalStatus.PARTIAL_SUCCESS
            orchestration.status = OrchestrationStatus.COMPLETED
        else:
            state.final_status = FinalStatus.FAILED
            orchestration.status = OrchestrationStatus.FAILED
            state.current_phase = Phase.FAILED

        orchestration.finished_at = time.time()
        await self._persist(orchestration)
        await self.event_bus.publish(
            orchestration.id,
            EventType.ORCHESTRATION_FINISHED,
            final_status=state.final_status.value,
            succeeded=succeeded,
            failed=failed,
            blocked=blocked,
            cancelled=cancelled,
            retries=retried,
            duration_seconds=duration,
        )

    # ── controls ────────────────────────────────────────────────────────────

    async def pause(self, orch_id: str) -> bool:
        scheduler = self._schedulers.get(orch_id)
        orchestration = await self.store.get_orchestration(orch_id)
        if orchestration is None:
            return False
        if scheduler is None:
            return False
        scheduler.pause()
        orchestration.status = OrchestrationStatus.PAUSED
        await self._persist(orchestration)
        return True

    async def resume(self, orch_id: str) -> bool:
        scheduler = self._schedulers.get(orch_id)
        orchestration = await self.store.get_orchestration(orch_id)
        if orchestration is None or scheduler is None:
            return False
        orchestration.pause_requested = False
        orchestration.status = OrchestrationStatus.RUNNING
        scheduler.resume()
        await self._persist(orchestration)
        return True

    async def cancel(self, orch_id: str) -> bool:
        orchestration = await self.store.get_orchestration(orch_id)
        if orchestration is None:
            return False
        orchestration.cancel_requested = True
        await self._persist(orchestration)
        scheduler = self._schedulers.get(orch_id)
        if scheduler is not None:
            scheduler.cancel()
            return True
        # Nothing running: mark it cancelled directly.
        if not orchestration.status.is_terminal:
            orchestration.status = OrchestrationStatus.CANCELLED
            orchestration.state.final_status = FinalStatus.CANCELLED
            orchestration.state.touch()
            orchestration.finished_at = time.time()
            await self._persist(orchestration)
        return True

    def is_running(self, orch_id: str) -> bool:
        job = self._jobs.get(orch_id)
        return job is not None and not job.done()

    async def wait(self, orch_id: str, timeout: float | None = None) -> bool:
        """Block until the run finishes. Returns False on timeout."""
        job = self._jobs.get(orch_id)
        if job is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(job), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def shutdown(self) -> None:
        """Cancel every live run. Called on app teardown."""
        for orch_id in list(self._schedulers):
            with contextlib.suppress(Exception):
                await self.cancel(orch_id)
        jobs = [j for j in self._jobs.values() if not j.done()]
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    # ── persistence ─────────────────────────────────────────────────────────

    async def _persist(self, orchestration: Orchestration) -> None:
        with contextlib.suppress(Exception):
            await self.store.save_orchestration(orchestration)

    async def get(self, orch_id: str) -> Optional[Orchestration]:
        return await self.store.get_orchestration(orch_id)

    async def list(self, project_id: str | None = None, limit: int = 100):
        return await self.store.list_orchestrations(project_id=project_id, limit=limit)

    # ── convenience ─────────────────────────────────────────────────────────

    @staticmethod
    def build_state(
        project_id: str, user_request: str, repository: str, dag
    ) -> ExecutionState:
        state = ExecutionState(
            project_id=project_id,
            original_task=user_request,
            repository=repository,
        )
        state.dag = dag
        state.tasks = dict(dag.tasks)
        state.total_tasks = len(dag.tasks)
        return state

    @staticmethod
    def progress_of(state: ExecutionState) -> dict[str, Any]:
        counts = {status.value: 0 for status in TaskStatus}
        for task in state.tasks.values():
            counts[task.status.value] += 1
        return counts

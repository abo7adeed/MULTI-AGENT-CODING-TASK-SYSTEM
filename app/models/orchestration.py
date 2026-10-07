"""
Orchestration model.

Wraps an `ExecutionState` with the lifecycle fields the API and the UI need:
run status, control flags (pause / resume / cancel), counters and a pointer to
the durable store row. A paused or cancelled run can be resumed because the
whole state is persisted, not held in a closure.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from app.models.domain import (
    ExecutionState,
    FinalStatus,
    OrchestrationStatus,
    new_id,
)


class Orchestration(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str = ""
    state: ExecutionState
    status: OrchestrationStatus = OrchestrationStatus.PENDING
    # control flags consumed by the ExecutionManager
    pause_requested: bool = False
    cancel_requested: bool = False
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    workspace_root: str = ""
    base_branch: str = "main"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    last_sequence: int = 0

    @property
    def final_status(self) -> FinalStatus:
        return self.state.final_status

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)

    @property
    def progress(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        done = sum(
            1
            for t in self.state.tasks.values()
            if t.status.value in ("SUCCESS", "FAILED", "CANCELLED")
        )
        return round(done / self.total_tasks, 4)

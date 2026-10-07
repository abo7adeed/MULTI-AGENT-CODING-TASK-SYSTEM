"""
In-process async event bus.

The orchestrator, scheduler, agents and integrator publish typed events here.
The API streams them to the browser over SSE, which is what makes the live
execution graph possible without polling.

The bus is intentionally simple: bounded per-subscriber queues with drop-oldest
on overflow. Durable event history lives in `app.models.store`.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Deque


class EventType(str, Enum):
    # lifecycle
    ORCHESTRATION_STARTED = "orchestration.started"
    ORCHESTRATION_PHASE = "orchestration.phase"
    ORCHESTRATION_FINISHED = "orchestration.finished"
    ORCHESTRATION_PAUSED = "orchestration.paused"
    ORCHESTRATION_RESUMED = "orchestration.resumed"
    ORCHESTRATION_CANCELLED = "orchestration.cancelled"
    ORCHESTRATION_LOG = "orchestration.log"

    # dag
    DAG_CREATED = "dag.created"
    DAG_UPDATED = "dag.updated"

    # tasks
    TASK_READY = "task.ready"
    TASK_STARTED = "task.started"
    TASK_SUCCEEDED = "task.succeeded"
    TASK_FAILED = "task.failed"
    TASK_RETRYING = "task.retrying"
    TASK_BLOCKED = "task.blocked"
    TASK_CANCELLED = "task.cancelled"

    # agents
    AGENT_STARTED = "agent.started"
    AGENT_FINISHED = "agent.finished"
    AGENT_LOG = "agent.log"

    # integration
    INTEGRATION_STARTED = "integration.started"
    CONFLICT_DETECTED = "conflict.detected"
    CONFLICT_RESOLVED = "conflict.resolved"
    TESTS_STARTED = "tests.started"
    TESTS_FINISHED = "tests.finished"
    INTEGRATION_FINISHED = "integration.finished"

    # generic
    HEARTBEAT = "heartbeat"


@dataclass(slots=True)
class Event:
    type: EventType
    orchestration_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "orchestration_id": self.orchestration_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "data": self.data,
        }


class Subscription:
    """An async iterator over events for one orchestration."""

    _SENTINEL = object()

    def __init__(self, bus: "EventBus", orchestration_id: str, replay: Deque[Event]):
        self._bus = bus
        self._orchestration_id = orchestration_id
        self._queue: asyncio.Queue = asyncio.Queue()
        for event in replay:
            self._queue.put_nowait(event)

    def push(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest so a slow consumer can never stall the run.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def close(self) -> None:
        self._queue.put_nowait(self._SENTINEL)

    async def __aiter__(self) -> AsyncIterator[Event]:
        try:
            while True:
                item = await self._queue.get()
                if item is self._SENTINEL:
                    return
                yield item  # type: ignore[misc]
        finally:
            self._bus.unsubscribe(self)


class EventBus:
    def __init__(self, history_size: int = 500, queue_size: int = 1000):
        self._subscribers: dict[str, list[Subscription]] = {}
        self._history: dict[str, Deque[Event]] = {}
        self._sequence: dict[str, int] = {}
        self._history_size = history_size
        self._queue_size = queue_size
        self._lock = asyncio.Lock()

    async def publish(
        self,
        orchestration_id: str,
        event_type: EventType,
        **data: Any,
    ) -> Event:
        async with self._lock:
            self._sequence[orchestration_id] = self._sequence.get(orchestration_id, 0) + 1
            event = Event(
                type=event_type,
                orchestration_id=orchestration_id,
                data=data,
                sequence=self._sequence[orchestration_id],
            )
            history = self._history.setdefault(orchestration_id, deque(maxlen=self._history_size))
            history.append(event)
            subscribers = list(self._subscribers.get(orchestration_id, ()))
        for sub in subscribers:
            sub.push(event)
        return event

    async def subscribe(self, orchestration_id: str) -> Subscription:
        async with self._lock:
            replay = deque(self._history.get(orchestration_id, ()))
            sub = Subscription(self, orchestration_id, replay)
            self._subscribers.setdefault(orchestration_id, []).append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        subs = self._subscribers.get(sub._orchestration_id)
        if subs and sub in subs:
            subs.remove(sub)

    def recent(
        self, orchestration_id: str, after_sequence: int = 0, limit: int = 200
    ) -> list[Event]:
        history = self._history.get(orchestration_id, ())
        return [e for e in history if e.sequence > after_sequence][-limit:]

    async def clear(self, orchestration_id: str) -> None:
        async with self._lock:
            self._history.pop(orchestration_id, None)
            self._sequence.pop(orchestration_id, None)
            for sub in list(self._subscribers.get(orchestration_id, ())):
                sub.close()
            self._subscribers.pop(orchestration_id, None)


_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def set_event_bus(bus: EventBus) -> None:
    global _bus
    _bus = bus

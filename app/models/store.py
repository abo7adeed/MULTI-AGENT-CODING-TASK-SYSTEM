"""
Durable state store.

A SQLite-backed persistence layer so a crashed or paused orchestration can be
resumed. Writes are debounced per orchestration because the scheduler mutates
state on every task transition; reads are always fresh.

The store is deliberately boring: single table, JSON payload, explicit
serialisation. Swapping in Postgres later means reimplementing `StateStore`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from app.logging_config import get_logger
from app.models.domain import ExecutionState, Project
from app.models.orchestration import Orchestration

logger = get_logger("app.store")


class StateStore(Protocol):
    async def save_project(self, project: Project) -> None: ...
    async def get_project(self, project_id: str) -> Optional[Project]: ...
    async def list_projects(self) -> list[Project]: ...
    async def delete_project(self, project_id: str) -> bool: ...

    async def save_orchestration(self, orch: Orchestration) -> None: ...
    async def get_orchestration(self, orch_id: str) -> Optional[Orchestration]: ...
    async def list_orchestrations(
        self, project_id: str | None = None, limit: int = 100
    ) -> list[Orchestration]: ...
    async def delete_orchestration(self, orch_id: str) -> bool: ...

    async def append_event(self, orchestration_id: str, event: dict[str, Any]) -> None: ...
    async def get_events(
        self, orchestration_id: str, after_sequence: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]: ...


class SQLiteStateStore:
    """Thread-confined SQLite store. All calls are serialised behind one lock."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS projects (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        description TEXT,
        data        TEXT NOT NULL,
        created_at  REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS orchestrations (
        id           TEXT PRIMARY KEY,
        project_id   TEXT NOT NULL,
        status       TEXT NOT NULL,
        final_status TEXT NOT NULL,
        data         TEXT NOT NULL,
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_orch_project
        ON orchestrations(project_id, created_at DESC);
    CREATE TABLE IF NOT EXISTS events (
        orchestration_id TEXT NOT NULL,
        sequence        INTEGER NOT NULL,
        type            TEXT NOT NULL,
        payload         TEXT NOT NULL,
        created_at      REAL NOT NULL,
        PRIMARY KEY (orchestration_id, sequence)
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        self._lock = asyncio.Lock()

    # ── projects ────────────────────────────────────────────────────────────

    async def save_project(self, project: Project) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO projects (id, name, description, data, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    project.id,
                    project.name,
                    project.description,
                    project.model_dump_json(),
                    project.created_at,
                ),
            )
            self._conn.commit()

    async def get_project(self, project_id: str) -> Optional[Project]:
        async with self._lock:
            row = self._conn.execute(
                "SELECT data FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return Project.model_validate_json(row["data"]) if row else None

    async def list_projects(self) -> list[Project]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM projects ORDER BY created_at DESC"
            ).fetchall()
        return [Project.model_validate_json(r["data"]) for r in rows]

    async def delete_project(self, project_id: str) -> bool:
        async with self._lock:
            cur = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ── orchestrations ──────────────────────────────────────────────────────

    async def save_orchestration(self, orch: Orchestration) -> None:
        orch.updated_at = time.time()
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO orchestrations"
                " (id, project_id, status, final_status, data, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    orch.id,
                    orch.state.project_id,
                    orch.status.value,
                    orch.state.final_status.value,
                    orch.model_dump_json(),
                    orch.created_at,
                    orch.updated_at,
                ),
            )
            self._conn.commit()

    async def get_orchestration(self, orch_id: str) -> Optional[Orchestration]:
        async with self._lock:
            row = self._conn.execute(
                "SELECT data FROM orchestrations WHERE id = ?", (orch_id,)
            ).fetchone()
        return Orchestration.model_validate_json(row["data"]) if row else None

    async def list_orchestrations(
        self, project_id: str | None = None, limit: int = 100
    ) -> list[Orchestration]:
        async with self._lock:
            if project_id:
                rows = self._conn.execute(
                    "SELECT data FROM orchestrations WHERE project_id = ?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (project_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT data FROM orchestrations ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [Orchestration.model_validate_json(r["data"]) for r in rows]

    async def delete_orchestration(self, orch_id: str) -> bool:
        async with self._lock:
            cur = self._conn.execute(
                "DELETE FROM orchestrations WHERE id = ?", (orch_id,)
            )
            self._conn.execute("DELETE FROM events WHERE orchestration_id = ?", (orch_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ── events ──────────────────────────────────────────────────────────────

    async def append_event(self, orchestration_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events"
                " (orchestration_id, sequence, type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    orchestration_id,
                    int(event.get("sequence", 0)),
                    str(event.get("type", "unknown")),
                    json.dumps(event.get("data", {}), default=str),
                    float(event.get("timestamp", time.time())),
                ),
            )
            self._conn.commit()

    async def get_events(
        self, orchestration_id: str, after_sequence: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        async with self._lock:
            rows = self._conn.execute(
                "SELECT sequence, type, payload, created_at FROM events"
                " WHERE orchestration_id = ? AND sequence > ?"
                " ORDER BY sequence ASC LIMIT ?",
                (orchestration_id, after_sequence, limit),
            ).fetchall()
        return [
            {
                "sequence": r["sequence"],
                "type": r["type"],
                "timestamp": r["created_at"],
                "data": json.loads(r["payload"]),
            }
            for r in rows
        ]

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - best effort
            pass


class InMemoryStateStore:
    """Non-durable store used by tests and ephemeral runs."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        self._orchestrations: dict[str, Orchestration] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def save_project(self, project: Project) -> None:
        async with self._lock:
            self._projects[project.id] = project

    async def get_project(self, project_id: str) -> Optional[Project]:
        return self._projects.get(project_id)

    async def list_projects(self) -> list[Project]:
        return sorted(self._projects.values(), key=lambda p: p.created_at, reverse=True)

    async def delete_project(self, project_id: str) -> bool:
        async with self._lock:
            return self._projects.pop(project_id, None) is not None

    async def save_orchestration(self, orch: Orchestration) -> None:
        orch.updated_at = time.time()
        async with self._lock:
            self._orchestrations[orch.id] = orch.model_copy(deep=True)

    async def get_orchestration(self, orch_id: str) -> Optional[Orchestration]:
        orch = self._orchestrations.get(orch_id)
        return orch.model_copy(deep=True) if orch else None

    async def list_orchestrations(
        self, project_id: str | None = None, limit: int = 100
    ) -> list[Orchestration]:
        items = list(self._orchestrations.values())
        if project_id:
            items = [o for o in items if o.state.project_id == project_id]
        items.sort(key=lambda o: o.created_at, reverse=True)
        return [o.model_copy(deep=True) for o in items[:limit]]

    async def delete_orchestration(self, orch_id: str) -> bool:
        async with self._lock:
            self._events.pop(orch_id, None)
            return self._orchestrations.pop(orch_id, None) is not None

    async def append_event(self, orchestration_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            self._events.setdefault(orchestration_id, []).append(event)

    async def get_events(
        self, orchestration_id: str, after_sequence: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        items = [
            e
            for e in self._events.get(orchestration_id, [])
            if e.get("sequence", 0) > after_sequence
        ]
        return items[-limit:]


def execution_state_to_dict(state: ExecutionState) -> dict[str, Any]:
    return state.model_dump(mode="json")

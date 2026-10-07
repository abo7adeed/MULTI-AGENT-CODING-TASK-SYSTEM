"""
Structured logging.

Emits newline-delimited JSON so log aggregation tools can index events, while
remaining readable in a terminal. A `task_id` / `orchestration_id` context
var keeps correlation across async tasks.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

_orchestration_id: ContextVar[str | None] = ContextVar("orchestration_id", default=None)
_task_id: ContextVar[str | None] = ContextVar("task_id", default=None)

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class ContextFilter(logging.Filter):
    """Injects the current orchestration/task ids into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.orchestration_id = _orchestration_id.get()
        record.task_id = _task_id.get()
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = self.formatTime(record, "%H:%M:%S")
        base = f"{color}{record.levelname:<8}{self.RESET} {ts} {record.name}: {record.getMessage()}"
        ctx = []
        if getattr(record, "orchestration_id", None):
            ctx.append(f"orch={record.orchestration_id[:8]}")
        if getattr(record, "task_id", None):
            ctx.append(f"task={record.task_id[:8]}")
        if ctx:
            base += "  [" + " ".join(ctx) + "]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_output else ConsoleFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Third-party noise
    for noisy in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── correlation context managers ─────────────────────────────────────────────


def orchestration_scope(orchestration_id: str):
    """Bind an orchestration id to the current context."""
    return _orchestration_id.set(orchestration_id)


def task_scope(task_id: str):
    """Bind a task id to the current context."""
    return _task_id.set(task_id)


def reset_orchestration_scope(token) -> None:
    _orchestration_id.reset(token)


def reset_task_scope(token) -> None:
    _task_id.reset(token)

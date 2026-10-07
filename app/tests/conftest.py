"""
Shared test fixtures.

Two invariants the whole suite depends on:

  * **No LLM.** Every test uses a mock provider or a mock agent, so the suite
    runs offline, deterministically and in CI without a key.
  * **No host repository.** Every test builds a throwaway git repo in tmp.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app.agents.mock import build_mock_roster
from app.agents.registry import AgentRegistry
from app.git.manager import GitManager


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    """An empty directory for a test repository."""
    path = tmp_path / "project"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def seed_repo(repo_path: Path) -> Path:
    """A minimal but realistic Python project, committed to git."""
    (repo_path / "app.py").write_text(
        "def main() -> str:\n    return 'base'\n", encoding="utf-8"
    )
    (repo_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    (repo_path / "tests").mkdir(exist_ok=True)
    (repo_path / "tests" / "test_app.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )
    return repo_path


@pytest.fixture
def git(seed_repo: Path) -> GitManager:
    """An initialised GitManager over a seeded repository."""
    manager = GitManager(seed_repo)

    async def setup() -> None:
        await manager.init_repo()
        await manager.commit_all(seed_repo, "Initial commit")

    asyncio.run(setup())
    return manager


@pytest.fixture
def empty_git(tmp_path: Path) -> GitManager:
    """A fresh, empty git repository (for worktree and merge tests)."""
    path = tmp_path / "empty"
    path.mkdir(parents=True, exist_ok=True)
    manager = GitManager(path)

    async def setup() -> None:
        await manager.init_repo()

    asyncio.run(setup())
    return manager


@pytest.fixture
def registry() -> AgentRegistry:
    """A registry with a mock agent for every role in the roster."""
    reg = AgentRegistry()
    reg.register_all(build_mock_roster(delay=0.0))
    return reg


@pytest.fixture
def workspace_root(tmp_path: Path) -> str:
    root = tmp_path / "workspaces"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch):
    """Settings pointed entirely at tmp, with fast deterministic policies."""
    from app.config import Settings

    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("MAX_PARALLEL_TASKS", "4")
    monkeypatch.setenv("MAX_TASK_RETRIES", "2")
    monkeypatch.setenv("RETRY_BACKOFF_SECONDS", "0.01")
    monkeypatch.setenv("RETRY_BACKOFF_MULTIPLIER", "1.0")
    monkeypatch.setenv("TASK_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("ORCHESTRATOR_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("TEST_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    return Settings(_env_file=None)


@pytest.fixture
def api_client(settings, registry, tmp_path, seed_repo):
    """A TestClient with an in-memory store and mock agents."""
    from fastapi.testclient import TestClient

    from app.api.deps import Container, set_container
    from app.api.main import create_app
    from app.config import get_settings
    from app.engine.execution import ExecutionManager
    from app.events import EventBus
    from app.llm.mock import MockLLMProvider
    from app.models.store import InMemoryStateStore

    get_settings.cache_clear()
    store = InMemoryStateStore()
    bus = EventBus()
    container = Container(
        settings=settings,
        store=store,  # type: ignore[arg-type]
        event_bus=bus,
        manager=ExecutionManager(store=store, event_bus=bus, settings=settings),
        registry=registry,
        provider=MockLLMProvider(),
        sandbox=__import__("app.sandbox.manager", fromlist=["NoSandbox"]).NoSandbox(),
        orchestrators={},
    )
    set_container(container)
    with TestClient(create_app()) as client:
        yield client
    set_container(None)
    get_settings.cache_clear()
    os.environ.pop("FREEBUFF_TEST", None)

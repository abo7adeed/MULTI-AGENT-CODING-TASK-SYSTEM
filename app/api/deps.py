"""
Application wiring.

One place that builds the object graph, so the API routes, the CLI and the
tests all get the same collaborators. Built lazily and cached, because the
agent registry needs an LLM provider and the store needs a path -- neither of
which should happen at import time.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

from app.agents.executor import build_default_registry
from app.agents.registry import AgentRegistry
from app.config import Settings, get_settings
from app.engine.execution import ExecutionManager
from app.events import EventBus, get_event_bus
from app.git.manager import GitManager
from app.llm.base import LLMProvider
from app.llm.factory import create_provider
from app.logging_config import configure_logging, get_logger
from app.models.store import SQLiteStateStore
from app.orchestrator.system import OrchestrationDeps, Orchestrator
from app.sandbox.manager import NoSandbox, build_sandbox

logger = get_logger("app.api.deps")


@dataclass
class Container:
    """The live object graph for one process."""

    settings: Settings
    store: SQLiteStateStore
    event_bus: EventBus
    manager: ExecutionManager
    registry: AgentRegistry
    provider: LLMProvider
    sandbox: object
    orchestrators: dict[str, Orchestrator] = None  # type: ignore[assignment]
    #: Result of the startup provider probe, cached so /health never has to
    #: make a network call of its own.
    provider_probe: Optional[dict] = None

    def orchestrator_for(
        self, repo_path: str, provider: Optional[LLMProvider] = None
    ) -> Orchestrator:
        """One orchestrator per repository path, reused across runs or custom per provider."""
        if provider is not None and provider is not self.provider:
            git = GitManager(str(repo_path))
            custom_registry = build_default_registry(provider=provider)
            return Orchestrator(
                OrchestrationDeps(
                    registry=custom_registry,
                    git=git,
                    llm=provider,
                    settings=self.settings,
                    event_bus=self.event_bus,
                    workspace_root=str(self.settings.workspaces_dir / key_hash(str(repo_path))),
                )
            )

        if self.orchestrators is None:
            self.orchestrators = {}
        key = str(repo_path)
        if key not in self.orchestrators:
            git = GitManager(key)
            self.orchestrators[key] = Orchestrator(
                OrchestrationDeps(
                    registry=self.registry,
                    git=git,
                    llm=self.provider,
                    settings=self.settings,
                    event_bus=self.event_bus,
                    workspace_root=str(self.settings.workspaces_dir / key_hash(key)),
                )
            )
        return self.orchestrators[key]

    def update_llm_provider(
        self,
        provider_name: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMProvider:
        """Dynamically switch the active LLM provider and rebuild the agent registry."""
        norm_name = provider_name.strip().lower()
        self.settings.llm_provider = norm_name
        if model is not None and model.strip():
            self.settings.llm_model = model.strip()

        if norm_name == "gemini":
            if api_key is not None:
                self.settings.gemini_api_key = api_key.strip()
            if model and model.strip():
                self.settings.gemini_model = model.strip()
            if base_url and base_url.strip():
                self.settings.gemini_base_url = base_url.strip()
        elif norm_name == "api":
            if api_key is not None:
                self.settings.api_key = api_key.strip()
            if model and model.strip():
                self.settings.api_model = model.strip()
            if base_url and base_url.strip():
                self.settings.api_base_url = base_url.strip()
        elif norm_name == "ollama":
            if api_key is not None:
                self.settings.ollama_api_key = api_key.strip()
            if model and model.strip():
                self.settings.ollama_model = model.strip()
            if base_url and base_url.strip():
                self.settings.ollama_base_url = base_url.strip()
        elif norm_name == "opencode":
            if model and model.strip():
                self.settings.opencode_model = model.strip()

        new_provider = create_provider(
            provider_name=self.settings.llm_provider,
            model=self.settings.llm_model or None,
            settings=self.settings,
        )
        self.provider = new_provider
        self.registry = build_default_registry(provider=new_provider)
        self.orchestrators = {}
        # The old verdict described the old provider.
        self.provider_probe = None
        return new_provider

    async def probe_provider(self) -> Optional[dict]:
        """
        Ask the active provider whether it can actually serve its model.

        Only providers that can answer the question get probed; the offline
        ones have nothing to be unreachable. Never raises -- a probe that blew
        up would be unable to report the failure it exists to find.
        """
        prober = getattr(self.provider, "probe", None)
        if prober is None:
            self.provider_probe = None
            return None
        try:
            self.provider_probe = await prober()
        except Exception as exc:  # noqa: BLE001
            self.provider_probe = {
                "provider": getattr(self.provider, "name", "unknown"),
                "model": getattr(self.provider, "model", ""),
                "reachable": False,
                "model_available": None,
                "models": [],
                "error": f"probe failed: {exc}",
            }
        return self.provider_probe

    async def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            await self.manager.shutdown()
        with contextlib.suppress(Exception):
            self.store.close()


def key_hash(path: str) -> str:
    import hashlib

    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def build_container(settings: Optional[Settings] = None) -> Container:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    store = SQLiteStateStore(settings.db_path)
    event_bus = get_event_bus()
    manager = ExecutionManager(store=store, event_bus=event_bus, settings=settings)

    provider = create_provider(settings=settings)
    registry = build_default_registry(provider=provider)
    sandbox = build_sandbox(
        settings, allowed_roots=[str(settings.workspaces_dir), settings.state_db_path]
    )

    logger.info(
        "Container ready",
        extra={
            "provider": provider.name,
            "model": provider.model,
            "roles": len(registry),
            "sandbox": type(sandbox).__name__,
            "db": settings.db_path,
        },
    )
    return Container(
        settings=settings,
        store=store,
        event_bus=event_bus,
        manager=manager,
        registry=registry,
        provider=provider,
        sandbox=sandbox,
        orchestrators={},
    )


_container: Optional[Container] = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = build_container()
    return _container


def set_container(container: Optional[Container]) -> None:
    """Inject a container. Used by tests to swap in an in-memory store."""
    global _container
    _container = container

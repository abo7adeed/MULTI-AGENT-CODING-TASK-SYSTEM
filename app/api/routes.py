"""
API routes.

Every endpoint is a thin adapter: validate, delegate to a collaborator, shape
the response. No orchestration logic lives here, which is what keeps the HTTP
layer swappable.

The streaming endpoint is the important one: the live execution graph in the UI
is driven by an SSE stream of the event bus, not by polling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import Container, get_container
from app.api.schemas import (
    AgentActivity,
    AgentInfo,
    AgentRetryRequest,
    DAGNode,
    DAGResponse,
    GitChange,
    HealthResponse,
    LogEntry,
    MessageResponse,
    OrchestrationCreate,
    OrchestrationDetail,
    OrchestrationSummary,
    ProjectCreate,
    ProjectResponse,
    SystemConfigUpdate,
    LLMTestRequest,
    LLMTestResponse,
    TaskCreate,
    TaskUpdate,
)
from app.engine.dag import DAGEngine
from app.engine.execution import ExecutionManager, RepositoryBusy
from app.events import EventType
from app.models.domain import (
    FinalStatus,
    OrchestrationStatus,
    Project,
    Task,
    TaskStatus,
)
from app.models.orchestration import Orchestration

router = APIRouter()
_STARTED_AT = time.time()


# ── helpers ──────────────────────────────────────────────────────────────────


async def _get_orchestration(container: Container, orch_id: str) -> Orchestration:
    found = await container.manager.get(orch_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Orchestration {orch_id} not found")
    return found


async def _active_run_on_repository(
    container: Container, repo_path: str
) -> Optional[Orchestration]:
    """
    A live run that already holds this repository, if there is one.

    A cheap pre-check, so a request that cannot be served fails before it
    spends model calls planning. The authoritative and race-free version of
    this check lives in `ExecutionManager.start`, under its lock.
    """
    wanted = ExecutionManager.normalise_repo(repo_path)
    if not wanted:
        return None
    for candidate in await container.store.list_orchestrations(limit=200):
        if candidate.status.is_terminal:
            continue
        if not container.manager.is_running(candidate.id):
            continue
        if ExecutionManager.normalise_repo(candidate.state.repository) == wanted:
            return candidate
    return None


def _summary(orch: Orchestration) -> OrchestrationSummary:
    return OrchestrationSummary(
        id=orch.id,
        name=orch.name,
        status=orch.status.value,
        final_status=orch.state.final_status.value,
        current_phase=orch.state.current_phase.value,
        total_tasks=orch.total_tasks,
        completed_tasks=orch.completed_tasks,
        failed_tasks=orch.failed_tasks,
        progress=orch.progress,
        duration_seconds=orch.duration_seconds,
        created_at=orch.created_at,
        started_at=orch.started_at,
        finished_at=orch.finished_at,
    )


def _detail(orch: Orchestration) -> OrchestrationDetail:
    base = _summary(orch).model_dump()
    state = orch.state
    return OrchestrationDetail(
        **base,
        project_id=state.project_id,
        original_task=state.original_task,
        repository=state.repository,
        tasks={tid: task.model_dump(mode="json") for tid, task in state.tasks.items()},
        agent_results={
            tid: result.model_dump(mode="json") for tid, result in state.agent_results.items()
        },
        errors=list(state.errors),
        test_results=state.test_results,
        repository_analysis=state.repository_analysis,
        task_analysis=state.task_analysis,
        integration_report=(
            state.integration_report.model_dump(mode="json")
            if state.integration_report
            else None
        ),
        updated_at=state.updated_at,
    )


# ── health & system ──────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    # Served from the cached startup probe: /health is polled by the UI, and it
    # must stay a fact about this process rather than a network round trip.
    probe = container.provider_probe or {}
    return HealthResponse(
        status="ok",
        provider=container.provider.name,
        model=container.provider.model,
        agents_registered=len(container.registry),
        sandbox=type(container.sandbox).__name__,
        database=container.settings.state_db_path,
        uptime_seconds=round(time.time() - _STARTED_AT, 1),
        provider_reachable=probe.get("reachable"),
        provider_model_available=probe.get("model_available"),
        provider_detail=probe.get("error") or "",
    )


def _env_file_path():
    """
    The `.env` the settings actually came from.

    `pydantic-settings` reads `./.env` relative to the process working
    directory, so that file wins when it exists. Falling back to the project's
    own file keeps a server started from elsewhere from writing a second copy
    of the config that nothing ever reads.
    """
    from pathlib import Path

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    return Path(__file__).resolve().parents[2] / ".env"


def _update_env_file(updates: dict[str, str]) -> None:
    from pathlib import Path
    env_path = _env_file_path()
    lines: list[str] = []
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
    existing_keys: dict[str, int] = {}
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if line_clean and not line_clean.startswith("#") and "=" in line_clean:
            k = line_clean.split("=", 1)[0].strip()
            existing_keys[k] = i
    for k, v in updates.items():
        if v is None:
            continue
        entry = f"{k}={v}"
        if k in existing_keys:
            lines[existing_keys[k]] = entry
        else:
            lines.append(entry)
    try:
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not persist settings to .env: %s", exc)


@router.get("/system/info", tags=["system"])
async def system_info(container: Container = Depends(get_container)) -> dict[str, Any]:
    from app.llm.factory import list_providers

    stats = container.provider.stats
    return {
        "llm": {
            "active_provider": container.provider.name,
            "active_model": container.provider.model,
            "available_providers": list_providers(),
            "stats": stats,
            "probe": container.provider_probe,
            "config": {
                "llm_provider": container.settings.llm_provider,
                "llm_model": container.settings.llm_model,
                "gemini_model": container.settings.gemini_model,
                "gemini_has_key": bool(container.settings.gemini_api_key),
                "api_model": container.settings.api_model,
                "api_base_url": container.settings.api_base_url,
                "api_has_key": bool(container.settings.api_key),
                "ollama_model": container.settings.ollama_model,
                "ollama_base_url": container.settings.ollama_base_url,
                "ollama_has_key": bool(container.settings.ollama_api_key),
                "ollama_num_predict": container.settings.ollama_num_predict,
                "opencode_model": container.settings.opencode_model,
            },
        },
        "scheduling": {
            "max_parallel_tasks": container.settings.max_parallel_tasks,
            "max_task_retries": container.settings.max_task_retries,
            "task_timeout_seconds": container.settings.task_timeout_seconds,
        },
        "sandbox": container.sandbox.describe(),
        "storage": {
            "database": container.settings.state_db_path,
            "workspace_root": container.settings.workspace_root,
        },
        "roster": [a.model_dump(mode="json") for a in container.registry.describe()],
    }


@router.post("/system/config", tags=["system"])
async def update_system_config(
    data: SystemConfigUpdate,
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """Dynamically switch the active LLM provider, model, API key or base URL."""
    try:
        new_provider = container.update_llm_provider(
            provider_name=data.llm_provider,
            model=data.llm_model,
            api_key=data.api_key,
            base_url=data.api_base_url,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to update LLM provider: {exc}") from exc

    if data.persist_to_env:
        env_map: dict[str, str] = {
            "LLM_PROVIDER": data.llm_provider,
        }
        if data.llm_model:
            env_map["LLM_MODEL"] = data.llm_model
        if data.llm_provider == "gemini":
            if data.api_key:
                env_map["GEMINI_API_KEY"] = data.api_key
            if data.llm_model:
                env_map["GEMINI_MODEL"] = data.llm_model
            if data.api_base_url:
                env_map["GEMINI_BASE_URL"] = data.api_base_url
        elif data.llm_provider == "api":
            if data.api_key:
                env_map["API_KEY"] = data.api_key
            if data.llm_model:
                env_map["API_MODEL"] = data.llm_model
            if data.api_base_url:
                env_map["API_BASE_URL"] = data.api_base_url
        elif data.llm_provider == "ollama":
            if data.api_key:
                env_map["OLLAMA_API_KEY"] = data.api_key
            if data.llm_model:
                env_map["OLLAMA_MODEL"] = data.llm_model
            if data.api_base_url:
                env_map["OLLAMA_BASE_URL"] = data.api_base_url
        elif data.llm_provider == "opencode":
            if data.llm_model:
                env_map["OPENCODE_MODEL"] = data.llm_model
        _update_env_file(env_map)

    return {
        "message": f"Switched to {new_provider.name} ({new_provider.model})",
        "provider": new_provider.name,
        "model": new_provider.model,
    }


@router.post("/system/test-llm", response_model=LLMTestResponse, tags=["system"])
async def test_llm_connection(
    data: LLMTestRequest,
    container: Container = Depends(get_container),
) -> LLMTestResponse:
    """Test connectivity and latency with a specific or active LLM provider."""
    from app.llm.factory import create_provider

    provider_name = (data.provider or container.settings.llm_provider).strip().lower()
    model = data.model or None
    kwargs: dict[str, Any] = {}
    if data.api_key is not None:
        # Ollama accepts a key too: the same provider serves a local server
        # (which ignores it) and ollama.com (which requires it).
        if provider_name in ("gemini", "api", "ollama"):
            kwargs["api_key"] = data.api_key
    if data.base_url:
        kwargs["base_url"] = data.base_url

    try:
        test_provider = create_provider(
            provider_name=provider_name,
            model=model,
            settings=container.settings,
            **kwargs,
        )
    except Exception as exc:
        return LLMTestResponse(
            success=False,
            provider=provider_name,
            model=model or "unknown",
            latency_ms=0.0,
            reply="",
            error=f"Could not initialize provider: {exc}",
        )

    prompt = data.prompt or "Hello! Please reply in one short sentence confirming you are working."
    t0 = time.time()
    try:
        response = await asyncio.wait_for(
            test_provider.complete(prompt, system_prompt="You are a helpful test agent."),
            timeout=25.0,
        )
        latency_ms = round((time.time() - t0) * 1000.0, 1)
        return LLMTestResponse(
            success=True,
            provider=test_provider.name,
            model=test_provider.model,
            latency_ms=latency_ms,
            reply=response.text[:500],
            error=None,
        )
    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000.0, 1)
        return LLMTestResponse(
            success=False,
            provider=test_provider.name,
            model=test_provider.model,
            latency_ms=latency_ms,
            reply="",
            error=str(exc)[:500],
        )


@router.get("/system/models", tags=["system"])
async def system_models(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Ask the configured provider which models it can actually reach."""
    provider = container.provider
    is_cloud = bool(getattr(provider, "is_cloud", False))
    base: dict[str, Any] = {
        "provider": provider.name,
        "model": provider.model,
        "cloud": is_cloud,
        "note": (
            "Ollama Cloud lists every hosted model, but a plan covers a subset of "
            "them. A model outside your plan is accepted here and fails with HTTP "
            "402 once a task runs, so prefer a model you have already used."
            if is_cloud
            else None
        ),
        "available": [],
        "error": None,
    }
    lister = getattr(provider, "list_available_models", None)
    if lister is not None:
        try:
            base["available"] = await asyncio.wait_for(lister(), timeout=20)
        except Exception as exc:  # noqa: BLE001
            base["error"] = str(exc)[:300]
    return base



# ── projects ─────────────────────────────────────────────────────────────────


@router.post("/projects", response_model=ProjectResponse, status_code=201, tags=["projects"])
async def create_project(
    data: ProjectCreate, container: Container = Depends(get_container)
) -> ProjectResponse:
    from pathlib import Path

    path = Path(data.local_path).expanduser()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())

    # The platform's isolation story depends on git worktrees, so a project
    # that is not a repository is initialised as one rather than failing later
    # with a confusing worktree error.
    from app.git.manager import GitManager

    git = GitManager(resolved)
    initialised = False
    if not await git.is_repo():
        try:
            await git.init_repo(data.base_branch or "main")
            initialised = True
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"could not initialise a git repository at {resolved}: {exc}",
            ) from exc

    project = Project(
        name=data.name,
        description=data.description,
        local_path=resolved,
        repository_url=data.repository_url,
        base_branch=await git.default_branch(),
    )
    await container.store.save_project(project)
    response = ProjectResponse(**project.model_dump(mode="json"))
    response.default_agent_profile = "default"
    return response


@router.get("/projects", response_model=list[ProjectResponse], tags=["projects"])
async def list_projects(container: Container = Depends(get_container)) -> list[ProjectResponse]:
    projects = await container.store.list_projects()
    out: list[ProjectResponse] = []
    for project in projects:
        orchestrations = await container.store.list_orchestrations(project_id=project.id)
        response = ProjectResponse(**project.model_dump(mode="json"))
        response.orchestration_count = len(orchestrations)
        response.running_count = sum(
            1 for o in orchestrations if o.status.value == "RUNNING"
        )
        out.append(response)
    return out


@router.get("/projects/{project_id}", response_model=ProjectResponse, tags=["projects"])
async def get_project(
    project_id: str, container: Container = Depends(get_container)
) -> ProjectResponse:
    project = await container.store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    orchestrations = await container.store.list_orchestrations(project_id=project_id)
    response = ProjectResponse(**project.model_dump(mode="json"))
    response.orchestration_count = len(orchestrations)
    response.running_count = sum(1 for o in orchestrations if o.status.value == "RUNNING")
    return response


@router.delete("/projects/{project_id}", response_model=MessageResponse, tags=["projects"])
async def delete_project(
    project_id: str, container: Container = Depends(get_container)
) -> MessageResponse:
    if not await container.store.delete_project(project_id):
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return MessageResponse(message=f"Project {project_id} deleted")


# ── tasks ────────────────────────────────────────────────────────────────────


@router.post(
    "/orchestrations/{orchestration_id}/tasks",
    response_model=Task,
    status_code=201,
    tags=["tasks"],
)
async def create_task(
    orchestration_id: str,
    data: TaskCreate,
    container: Container = Depends(get_container),
) -> Task:
    """Add a task to a run. Refused once the run is terminal."""
    orchestration = await _get_orchestration(container, orchestration_id)
    if orchestration.status.is_terminal:
        raise HTTPException(
            status_code=409, detail="cannot add tasks to a finished orchestration"
        )
    task = Task(
        title=data.title,
        description=data.description,
        type=data.type,
        priority=data.priority,
        dependencies=data.dependencies,
        assigned_agent=data.assigned_agent,
        max_retries=data.max_retries,
    )
    orchestration.state.dag.add(task)
    orchestration.state.tasks[task.id] = task
    orchestration.total_tasks = len(orchestration.state.dag.tasks)
    await container.store.save_orchestration(orchestration)
    await container.event_bus.publish(
        orchestration_id, EventType.DAG_UPDATED, reason=f"task {task.id} added"
    )
    return task


@router.get("/orchestrations/{orchestration_id}/tasks/{task_id}", response_model=Task, tags=["tasks"])
@router.get("/tasks/{task_id}", response_model=Task, tags=["tasks"])
async def get_task(
    task_id: str,
    orchestration_id: Optional[str] = None,
    container: Container = Depends(get_container),
) -> Task:
    """Fetch a task by id, searching the given orchestration or all of them."""
    if orchestration_id:
        orchestration = await _get_orchestration(container, orchestration_id)
        task = orchestration.state.tasks.get(task_id)
    else:
        task = None
        for candidate in await container.store.list_orchestrations(limit=200):
            if task_id in candidate.state.tasks:
                task = candidate.state.tasks[task_id]
                break
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.patch(
    "/orchestrations/{orchestration_id}/tasks/{task_id}", response_model=Task, tags=["tasks"]
)
async def update_task(
    orchestration_id: str,
    task_id: str,
    data: TaskUpdate,
    container: Container = Depends(get_container),
) -> Task:
    orchestration = await _get_orchestration(container, orchestration_id)
    task = orchestration.state.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status in (TaskStatus.RUNNING, TaskStatus.SUCCESS):
        raise HTTPException(
            status_code=409,
            detail=f"cannot edit a task that is {task.status.value.lower()}",
        )
    updates = data.model_dump(exclude_none=True)
    for key, value in updates.items():
        setattr(task, key, value)
    orchestration.state.dag.rebuild_edges()
    await container.store.save_orchestration(orchestration)
    return task


# ── orchestrations ───────────────────────────────────────────────────────────


@router.get("/orchestrations", response_model=list[OrchestrationSummary], tags=["orchestrations"])
async def list_orchestrations(
    project_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    container: Container = Depends(get_container),
) -> list[OrchestrationSummary]:
    found = await container.store.list_orchestrations(project_id=project_id, limit=limit)
    return [_summary(o) for o in found]


@router.post(
    "/orchestrations",
    response_model=OrchestrationDetail,
    status_code=201,
    tags=["orchestrations"],
)
async def create_orchestration(
    data: OrchestrationCreate,
    container: Container = Depends(get_container),
) -> OrchestrationDetail:
    """
    Plan a run, persist it, and launch it in the background.

    Planning happens synchronously so the caller immediately receives the DAG;
    execution continues on a background task and is observable over SSE.
    """
    project = await container.store.get_project(data.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {data.project_id} not found")

    active = await _active_run_on_repository(container, project.local_path)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"orchestration {active.id} is already running on {project.local_path}. "
                "Agents share one repository's worktrees and branches, so a second "
                "concurrent run would corrupt both. Wait for it to finish, or cancel it "
                f"with POST /orchestrations/{active.id}/cancel."
            ),
        )

    custom_provider = None
    if data.llm_provider:
        from app.llm.factory import create_provider
        kwargs: dict[str, Any] = {}
        if data.api_key:
            kwargs["api_key"] = data.api_key
        if data.api_base_url:
            kwargs["base_url"] = data.api_base_url
        custom_provider = create_provider(
            provider_name=data.llm_provider,
            model=data.llm_model,
            settings=container.settings,
            **kwargs,
        )

    orchestrator = container.orchestrator_for(project.local_path, provider=custom_provider)
    orchestration = await orchestrator.plan(project, data.user_request)

    if data.name:
        orchestration.name = data.name
    orchestration.total_tasks = len(orchestration.state.dag.tasks)
    await container.store.save_orchestration(orchestration)

    try:
        await container.manager.start(
            orchestration, orchestrator.bind(orchestration, container.manager)
        )
    except RepositoryBusy as exc:
        # The pre-check above can be overtaken by a run that started while this
        # one was planning; this check cannot be. The plan was already persisted
        # by then, so remove it: a run that was refused never existed, and
        # leaving it behind would show up as a phantom PENDING entry forever.
        with contextlib.suppress(Exception):
            await container.store.delete_orchestration(orchestration.id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if data.wait:
        await container.manager.wait(
            orchestration.id, timeout=container.settings.orchestrator_timeout_seconds
        )
        orchestration = await _get_orchestration(container, orchestration.id)

    return _detail(orchestration)


@router.get(
    "/orchestrations/{orchestration_id}",
    response_model=OrchestrationDetail,
    tags=["orchestrations"],
)
async def get_orchestration(
    orchestration_id: str, container: Container = Depends(get_container)
) -> OrchestrationDetail:
    return _detail(await _get_orchestration(container, orchestration_id))


@router.delete(
    "/orchestrations/{orchestration_id}", response_model=MessageResponse, tags=["orchestrations"]
)
async def delete_orchestration(
    orchestration_id: str, container: Container = Depends(get_container)
) -> MessageResponse:
    await container.manager.cancel(orchestration_id)
    if not await container.store.delete_orchestration(orchestration_id):
        raise HTTPException(status_code=404, detail=f"Orchestration {orchestration_id} not found")
    await container.event_bus.clear(orchestration_id)
    return MessageResponse(message=f"Orchestration {orchestration_id} deleted")


@router.post(
    "/orchestrations/{orchestration_id}/pause", response_model=MessageResponse, tags=["orchestrations"]
)
async def pause_orchestration(
    orchestration_id: str, container: Container = Depends(get_container)
) -> MessageResponse:
    if not await container.manager.pause(orchestration_id):
        raise HTTPException(
            status_code=409, detail="orchestration is not running and cannot be paused"
        )
    return MessageResponse(message="Paused; in-flight agents will finish first")


@router.post(
    "/orchestrations/{orchestration_id}/resume", response_model=MessageResponse, tags=["orchestrations"]
)
async def resume_orchestration(
    orchestration_id: str, container: Container = Depends(get_container)
) -> MessageResponse:
    if not await container.manager.resume(orchestration_id):
        raise HTTPException(
            status_code=409, detail="orchestration is not paused and cannot be resumed"
        )
    return MessageResponse(message="Resumed")


@router.post(
    "/orchestrations/{orchestration_id}/cancel", response_model=MessageResponse, tags=["orchestrations"]
)
async def cancel_orchestration(
    orchestration_id: str, container: Container = Depends(get_container)
) -> MessageResponse:
    if not await container.manager.cancel(orchestration_id):
        raise HTTPException(status_code=404, detail=f"Orchestration {orchestration_id} not found")
    return MessageResponse(message="Cancellation requested; running agents were stopped")


# ── dag ──────────────────────────────────────────────────────────────────────


@router.get(
    "/orchestrations/{orchestration_id}/dag", response_model=DAGResponse, tags=["orchestrations"]
)
async def get_dag(
    orchestration_id: str, container: Container = Depends(get_container)
) -> DAGResponse:
    """The full execution graph: nodes with live status, plus the wave layout."""
    orchestration = await _get_orchestration(container, orchestration_id)
    state = orchestration.state
    engine = DAGEngine(state.dag)

    waves = engine.parallel_waves() if state.dag.tasks else []
    wave_of: dict[str, int] = {}
    for index, wave in enumerate(waves):
        for task_id in wave:
            wave_of[task_id] = index

    depth_of: dict[str, int] = {}
    for task_id in engine.topological_order():
        task = state.dag.tasks[task_id]
        depth_of[task_id] = (
            1 + max((depth_of.get(d, 0) for d in task.dependencies), default=-1)
            if task.dependencies
            else 0
        )

    nodes: list[DAGNode] = []
    edges: list[dict[str, str]] = []
    for task_id, task in state.dag.tasks.items():
        result = state.agent_results.get(task_id)
        nodes.append(
            DAGNode(
                id=task_id,
                title=task.title,
                type=task.type.value,
                status=task.status.value,
                priority=task.priority,
                agent=task.assigned_agent,
                dependencies=task.dependencies,
                wave=wave_of.get(task_id, 0),
                depth=depth_of.get(task_id, 0),
                duration_seconds=task.duration_seconds,
                retry_count=task.retry_count,
                files_changed=result.files_changed if result else [],
                tests=task.tests,
                errors=task.errors,
                blocked_reason=task.blocked_reason,
                commit=result.commit if result else None,
            )
        )
        for dep in task.dependencies:
            edges.append({"from": dep, "to": task_id})

    summary = engine.summary() if state.dag.tasks else {}
    return DAGResponse(
        orchestration_id=orchestration_id,
        total_tasks=len(state.dag.tasks),
        waves=waves,
        critical_path=engine.critical_path() if state.dag.tasks else [],
        max_parallelism=summary.get("max_wave_size", 0),
        status_counts=summary.get("status_counts", {}),
        nodes=sorted(nodes, key=lambda n: (n.wave, n.priority, n.title)),
        edges=edges,
    )


# ── agents ───────────────────────────────────────────────────────────────────


@router.get("/agents", response_model=list[AgentInfo], tags=["agents"])
async def list_agents(container: Container = Depends(get_container)) -> list[AgentInfo]:
    return [AgentInfo(**a.model_dump(mode="json")) for a in container.registry.describe()]


@router.get(
    "/orchestrations/{orchestration_id}/agents",
    response_model=list[AgentActivity],
    tags=["agents"],
)
async def orchestration_agents(
    orchestration_id: str, container: Container = Depends(get_container)
) -> list[AgentActivity]:
    orchestration = await _get_orchestration(container, orchestration_id)
    state = orchestration.state
    activity: list[AgentActivity] = []
    for task_id, task in state.tasks.items():
        result = state.agent_results.get(task_id)
        activity.append(
            AgentActivity(
                task_id=task_id,
                title=task.title,
                role=task.assigned_agent or (result.agent_role if result else None),
                status=(result.status if result else task.status).value,
                summary=result.summary if result else "",
                files_changed=result.files_changed if result else [],
                commit=result.commit if result else None,
                branch=task.branch,
                duration_seconds=task.duration_seconds,
                attempt=result.attempt if result else 1,
                errors=task.errors or (result.errors if result else []),
                recommendations=result.recommendations if result else [],
                tests_passed=result.tests_passed if result else 0,
                tests_failed=result.tests_failed if result else 0,
            )
        )
    order = {s.value: i for i, s in enumerate(TaskStatus)}
    activity.sort(key=lambda a: (order.get(a.status, 9), a.title))
    return activity


@router.post(
    "/agents/{task_id}/retry", response_model=MessageResponse, tags=["agents"]
)
async def retry_agent(
    task_id: str,
    data: AgentRetryRequest = Body(default=AgentRetryRequest()),
    orchestration_id: Optional[str] = None,
    container: Container = Depends(get_container),
) -> MessageResponse:
    """
    Re-queue a failed or blocked task.

    Works both for a run that is still live and for one that already finished:
    in the latter case a fresh orchestration is started from the reset state.
    """
    orchestration = (
        await _get_orchestration(container, orchestration_id)
        if orchestration_id
        else await _find_orchestration_with_task(container, task_id)
    )
    if orchestration is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = orchestration.state.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail="task is currently running")

    task.status = TaskStatus.READY
    task.retry_count = 0
    task.errors = []
    task.output.pop("last_error", None)
    task.blocked_reason = None
    orchestration.state.agent_results.pop(task_id, None)

    if data.reset_dependencies:
        from app.engine.dag import DAGEngine

        engine = DAGEngine(orchestration.state.dag)
        for dependent_id in engine.descendants(task_id):
            dependent = orchestration.state.tasks.get(dependent_id)
            if dependent and dependent.status == TaskStatus.BLOCKED:
                dependent.status = TaskStatus.READY
                dependent.blocked_reason = None

    orchestration.state.errors = [
        e for e in orchestration.state.errors if task_id not in e
    ]

    orchestrator = container.orchestrator_for(orchestration.state.repository)
    if orchestration.status.is_terminal:
        orchestration.status = OrchestrationStatus.RUNNING
        orchestration.finished_at = None
        orchestration.state.final_status = FinalStatus.RUNNING
        await container.manager.start(
            orchestration, orchestrator.bind(orchestration, container.manager)
        )
    else:
        await container.store.save_orchestration(orchestration)
        await container.event_bus.publish(
            orchestration.id, EventType.TASK_READY, task_id=task_id, title=task.title
        )

    return MessageResponse(
        message=f"Task {task.title} queued for retry",
        detail=data.reason or None,
    )


async def _find_orchestration_with_task(
    container: Container, task_id: str
) -> Optional[Orchestration]:
    for candidate in await container.store.list_orchestrations(limit=200):
        if task_id in candidate.state.tasks:
            return candidate
    return None


# ── logs & events ────────────────────────────────────────────────────────────


@router.get(
    "/orchestrations/{orchestration_id}/logs",
    response_model=list[LogEntry],
    tags=["logs"],
)
async def get_logs(
    orchestration_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
    container: Container = Depends(get_container),
) -> list[LogEntry]:
    await _get_orchestration(container, orchestration_id)
    # Live events first (they carry the most recent detail), then the
    # durable history for anything that has already scrolled out of memory.
    live = [
        LogEntry(sequence=e.sequence, type=e.type.value, timestamp=e.timestamp, data=e.data)
        for e in container.event_bus.recent(orchestration_id, after, limit)
    ]
    seen = {entry.sequence for entry in live}
    stored = [
        LogEntry(sequence=e["sequence"], type=e["type"], timestamp=e["timestamp"], data=e["data"])
        for e in await container.store.get_events(orchestration_id, after, limit)
        if e["sequence"] not in seen
    ]
    return sorted(live + stored, key=lambda e: e.sequence)[:limit]


@router.get("/orchestrations/{orchestration_id}/stream", tags=["logs"])
async def stream_events(
    orchestration_id: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    container: Container = Depends(get_container),
) -> StreamingResponse:
    """
    Server-Sent Events stream of the run.

    Replays everything after `after` from the in-memory bus, then follows live.
    Closes itself when the run reaches a terminal state so the browser stops
    reconnecting.
    """
    await _get_orchestration(container, orchestration_id)

    async def generator():
        subscription = await container.event_bus.subscribe(orchestration_id)
        try:
            yield f"event: open\ndata: {json.dumps({'orchestration_id': orchestration_id})}\n\n"
            async for event in subscription:
                if event.sequence and event.sequence <= after:
                    continue
                yield (
                    f"id: {event.sequence}\n"
                    f"event: {event.type.value}\n"
                    f"data: {json.dumps(event.to_dict(), default=str)}\n\n"
                )
                if event.type in (
                    EventType.ORCHESTRATION_FINISHED,
                    EventType.ORCHESTRATION_CANCELLED,
                ):
                    break
                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:  # client went away
            raise
        finally:
            container.event_bus.unsubscribe(subscription)
            # Tell the client this stream is done so EventSource closes cleanly.
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── git changes & results ────────────────────────────────────────────────────


@router.get(
    "/orchestrations/{orchestration_id}/changes",
    response_model=list[GitChange],
    tags=["git"],
)
async def get_changes(
    orchestration_id: str, container: Container = Depends(get_container)
) -> list[GitChange]:
    orchestration = await _get_orchestration(container, orchestration_id)
    state = orchestration.state
    changes: list[GitChange] = []
    for task_id, result in state.agent_results.items():
        task = state.tasks.get(task_id)
        changes.append(
            GitChange(
                branch=result.branch or (task.branch if task else None),
                commit=result.commit,
                task_id=task_id,
                title=task.title if task else task_id[:8],
                files_changed=result.files_changed,
            )
        )
    return changes


@router.get(
    "/orchestrations/{orchestration_id}/diff", tags=["git"]
)
async def get_diff(
    orchestration_id: str,
    ref: Optional[str] = None,
    stat_only: bool = True,
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """The integrated diff, or the diff of one task's branch against base."""
    orchestration = await _get_orchestration(container, orchestration_id)
    from app.git.manager import GitManager

    git = GitManager(orchestration.state.repository)
    if not await git.is_repo():
        raise HTTPException(status_code=400, detail="project is not a git repository")

    branch = ref
    if branch is None:
        report = orchestration.state.integration_report
        base = orchestration.state.created_at
        branch = report.merged_branches[0] if (report and report.merged_branches) else None
    if branch is None:
        status = await git.status()
        return {"files": status, "patch": "", "stat_only": stat_only}

    base_ref = orchestration.base_branch or await git.default_branch()
    try:
        diff = await git.diff(base_ref, branch, stat_only=stat_only)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not diff: {exc}") from exc
    return {
        "branch": branch,
        "base": base_ref,
        "files": diff.files_changed,
        "insertions": diff.insertions,
        "deletions": diff.deletions,
        "patch": diff.patch,
        "stat_only": stat_only,
    }


@router.get(
    "/orchestrations/{orchestration_id}/test-results", tags=["results"]
)
async def get_test_results(
    orchestration_id: str, container: Container = Depends(get_container)
) -> dict[str, Any]:
    orchestration = await _get_orchestration(container, orchestration_id)
    state = orchestration.state
    return {
        "summary": state.test_results,
        "runs": [
            run.model_dump(mode="json")
            for run in (state.integration_report.test_runs if state.integration_report else [])
        ],
        "report": state.integration_report.report if state.integration_report else "",
    }


@router.get(
    "/orchestrations/{orchestration_id}/report", tags=["results"]
)
async def get_report(orchestration_id: str, container: Container = Depends(get_container)):
    """The final integration report, rendered as markdown."""
    orchestration = await _get_orchestration(container, orchestration_id)
    report = orchestration.state.integration_report
    if report is None:
        raise HTTPException(
            status_code=404, detail="integration has not run for this orchestration"
        )
    return {
        "markdown": report.report,
        "success": report.success,
        "merged_branches": report.merged_branches,
        "skipped": report.skipped_details,
        "conflicts": report.conflicts,
        "regressions": report.regressions,
        "review_findings": report.review_findings,
    }

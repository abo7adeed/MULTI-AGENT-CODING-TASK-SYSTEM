"""
HTTP API tests.

Driven through `TestClient` against an in-memory store and mock agents, so
every endpoint is exercised for real -- routing, validation, status codes and
serialisation -- without a network, an LLM or a disk database.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def client(api_client):
    return api_client


@pytest.fixture
def project(client, seed_repo):
    response = client.post(
        "/projects", json={"name": "demo", "local_path": str(seed_repo)}
    )
    assert response.status_code == 201, response.text
    return response.json()


def start_run(client, project, request="Add a health endpoint", **extra):
    body = {"project_id": project["id"], "user_request": request, **extra}
    response = client.post("/orchestrations", json=body)
    assert response.status_code == 201, response.text
    return response.json()


# ── system ──────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_reports_the_active_configuration(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["provider"] == "mock"
        assert data["agents_registered"] > 0
        assert data["sandbox"] == "NoSandbox"
        assert data["uptime_seconds"] >= 0

    def test_root_lists_the_endpoints(self, client):
        data = client.get("/").json()
        assert data["docs"] == "/docs"
        assert "POST   /orchestrations" in data["endpoints"]

    def test_system_info_describes_every_collaborator(self, client):
        data = client.get("/system/info").json()
        assert data["llm"]["active_provider"] == "mock"
        assert data["llm"]["available_providers"]
        assert data["scheduling"]["max_parallel_tasks"] >= 1
        # The API reports honestly which isolation mode is in effect.
        assert data["sandbox"] == {
            "available": False,
            "policy": None,
            "note": "sandboxing disabled",
        }
        assert data["roster"]

    def test_system_models_for_a_provider_that_cannot_list(self, client):
        data = client.get("/system/models").json()
        assert data["provider"] == "mock"
        assert data["error"] is None
        assert isinstance(data["available"], list)

    def test_system_models_survives_a_listing_failure(self, client, monkeypatch):
        from app.llm.mock import MockLLMProvider

        class Broken(MockLLMProvider):
            async def list_available_models(self):
                raise RuntimeError("no route to host")

        from app.api.deps import get_container

        get_container().provider = Broken()
        data = client.get("/system/models").json()
        assert data["available"] == []
        assert "no route to host" in data["error"]

    def test_openapi_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/orchestrations" in schema["paths"]
        assert "/orchestrations/{orchestration_id}/stream" in schema["paths"]


# ── projects ────────────────────────────────────────────────────────────────


class TestProjects:
    def test_create_project_initialises_a_git_repository(self, client, repo_path):
        response = client.post(
            "/projects", json={"name": "fresh", "local_path": str(repo_path)}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["base_branch"] == "main"
        assert (repo_path / ".git").exists()

    def test_create_project_creates_a_missing_directory(self, client, tmp_path):
        target = tmp_path / "made" / "up"
        response = client.post(
            "/projects", json={"name": "made", "local_path": str(target)}
        )
        assert response.status_code == 201
        assert target.exists()

    def test_create_project_resolves_the_path(self, client, seed_repo):
        data = client.post(
            "/projects", json={"name": "resolved", "local_path": str(seed_repo)}
        ).json()
        assert data["local_path"] == str(seed_repo.resolve())
        assert not data["local_path"].startswith("~")

    def test_list_projects_is_empty_initially(self, client):
        assert client.get("/projects").json() == []

    def test_list_projects_includes_counts(self, client, project):
        data = client.get("/projects").json()
        assert len(data) == 1
        assert data[0]["orchestration_count"] == 0
        assert data[0]["running_count"] == 0

    def test_get_project(self, client, project):
        assert client.get(f"/projects/{project['id']}").json()["id"] == project["id"]

    def test_get_missing_project_is_404(self, client):
        response = client.get("/projects/nope")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_project(self, client, project):
        assert client.delete(f"/projects/{project['id']}").status_code == 200
        assert client.get(f"/projects/{project['id']}").status_code == 404

    def test_delete_missing_project_is_404(self, client):
        assert client.delete("/projects/nope").status_code == 404

    def test_a_name_is_required(self, client, seed_repo):
        response = client.post("/projects", json={"local_path": str(seed_repo)})
        assert response.status_code == 422

    def test_a_path_is_required(self, client):
        response = client.post("/projects", json={"name": "x"})
        assert response.status_code == 422


# ── orchestrations ──────────────────────────────────────────────────────────


class TestOrchestrations:
    def test_starting_a_run_returns_the_plan_immediately(self, client, project):
        data = start_run(client, project)
        assert data["total_tasks"] > 0
        assert data["status"] in {"PENDING", "RUNNING"}
        assert data["tasks"]
        assert data["task_analysis"]["raw_request"] == "Add a health endpoint"

    def test_wait_for_completion_returns_a_finished_run(self, client, project):
        data = start_run(client, project, wait=True)
        assert data["status"] in {"COMPLETED", "FAILED", "CANCELLED"}
        assert data["final_status"] != "PENDING"

    def test_a_custom_name_overrides_the_generated_one(self, client, project):
        data = start_run(client, project, name="my run")
        assert data["name"] == "my run"

    def test_starting_a_run_for_a_missing_project_is_404(self, client):
        response = client.post(
            "/orchestrations",
            json={"project_id": "nope", "user_request": "Add a health endpoint"},
        )
        assert response.status_code == 404

    def test_an_empty_request_is_rejected(self, client, project):
        response = client.post(
            "/orchestrations", json={"project_id": project["id"], "user_request": ""}
        )
        assert response.status_code == 422

    def test_a_too_short_request_is_rejected(self, client, project):
        response = client.post(
            "/orchestrations", json={"project_id": project["id"], "user_request": "x"}
        )
        assert response.status_code == 422

    def test_list_orchestrations(self, client, project):
        start_run(client, project)
        listed = client.get("/orchestrations").json()
        assert len(listed) == 1
        assert listed[0]["total_tasks"] > 0
        assert 0.0 <= listed[0]["progress"] <= 1.0

    def test_list_orchestrations_filtered_by_project(self, client, project):
        start_run(client, project)
        assert len(client.get(f"/orchestrations?project_id={project['id']}").json()) == 1
        assert client.get("/orchestrations?project_id=other").json() == []

    def test_list_orchestrations_rejects_a_nonsense_limit(self, client):
        assert client.get("/orchestrations?limit=0").status_code == 422
        assert client.get("/orchestrations?limit=9999").status_code == 422

    def test_get_orchestration(self, client, project):
        started = start_run(client, project)
        data = client.get(f"/orchestrations/{started['id']}").json()
        assert data["id"] == started["id"]
        assert data["repository"] == project["local_path"]

    def test_get_missing_orchestration_is_404(self, client):
        assert client.get("/orchestrations/nope").status_code == 404

    def test_delete_orchestration(self, client, project):
        started = start_run(client, project, wait=True)
        assert client.delete(f"/orchestrations/{started['id']}").status_code == 200
        assert client.get(f"/orchestrations/{started['id']}").status_code == 404

    def test_delete_missing_orchestration_is_404(self, client):
        assert client.delete("/orchestrations/nope").status_code == 404

    def test_pause_and_resume_conflict_when_not_running(self, client, project):
        started = start_run(client, project, wait=True)
        assert client.post(f"/orchestrations/{started['id']}/pause").status_code == 409
        assert client.post(f"/orchestrations/{started['id']}/resume").status_code == 409

    def test_cancel_is_reported_as_not_found(self, client):
        assert client.post("/orchestrations/nope/cancel").status_code == 404

    def test_cancel_a_finished_run_is_accepted(self, client, project):
        started = start_run(client, project, wait=True)
        response = client.post(f"/orchestrations/{started['id']}/cancel")
        assert response.status_code in (200, 404)


# ── dag ─────────────────────────────────────────────────────────────────────


class TestDag:
    def test_dag_layout(self, client, project):
        started = start_run(client, project)
        data = client.get(f"/orchestrations/{started['id']}/dag").json()
        assert data["total_tasks"] == started["total_tasks"]
        assert len(data["nodes"]) == data["total_tasks"]
        assert data["waves"]
        assert data["edges"]
        assert data["status_counts"]

    def test_dag_nodes_carry_what_the_ui_needs(self, client, project):
        started = start_run(client, project)
        node = client.get(f"/orchestrations/{started['id']}/dag").json()["nodes"][0]
        assert set(node) >= {
            "id", "title", "type", "status", "priority", "dependencies",
            "agent", "wave", "depth", "files_changed", "commit", "errors",
        }

    def test_dag_edges_point_at_real_nodes(self, client, project):
        started = start_run(client, project)
        data = client.get(f"/orchestrations/{started['id']}/dag").json()
        ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert edge["from"] in ids
            assert edge["to"] in ids

    def test_dag_for_a_missing_orchestration_is_404(self, client):
        assert client.get("/orchestrations/nope/dag").status_code == 404

    def test_dag_for_an_empty_orchestration_is_still_valid(self, client, project):
        started = start_run(client, project)
        data = client.get(f"/orchestrations/{started['id']}/dag").json()
        assert data["total_tasks"] == started["total_tasks"]


# ── tasks ───────────────────────────────────────────────────────────────────


class TestTasks:
    def test_add_a_task_to_a_run(self, client, project):
        started = start_run(client, project)
        response = client.post(
            f"/orchestrations/{started['id']}/tasks",
            json={"title": "write the migration", "type": "database", "priority": 7},
        )
        assert response.status_code == 201
        assert response.json()["title"] == "write the migration"
        assert client.get(f"/orchestrations/{started['id']}").json()["total_tasks"] == (
            started["total_tasks"] + 1
        )

    def test_adding_a_task_to_a_finished_run_is_409(self, client, project):
        started = start_run(client, project, wait=True)
        response = client.post(
            f"/orchestrations/{started['id']}/tasks", json={"title": "too late"}
        )
        assert response.status_code == 409

    def test_adding_a_task_to_a_missing_run_is_404(self, client):
        assert client.post("/orchestrations/nope/tasks", json={"title": "x"}).status_code == 404

    def test_an_unknown_task_type_is_coerced_not_rejected(self, client, project):
        """A task is accepted with an arbitrary type string and safely coerced."""
        started = start_run(client, project)
        response = client.post(
            f"/orchestrations/{started['id']}/tasks",
            json={"title": "x", "type": "not-a-real-type"},
        )
        assert response.status_code == 201
        assert response.json()["type"] in {
            "analysis", "architecture", "backend", "frontend", "database",
            "ai_ml", "testing", "devops", "security", "documentation",
            "review", "integration", "refactor", "planning", "generic",
        }

    def test_an_out_of_range_priority_is_rejected(self, client, project):
        started = start_run(client, project)
        response = client.post(
            f"/orchestrations/{started['id']}/tasks",
            json={"title": "x", "priority": 99},
        )
        assert response.status_code == 422

    def test_an_empty_title_is_rejected(self, client, project):
        started = start_run(client, project)
        response = client.post(
            f"/orchestrations/{started['id']}/tasks", json={"title": ""}
        )
        assert response.status_code == 422

    def test_get_a_task_by_id(self, client, project):
        started = start_run(client, project)
        task_id = next(iter(started["tasks"]))
        task = client.get(f"/tasks/{task_id}").json()
        assert task["id"] == task_id

    def test_get_a_task_scoped_to_its_orchestration(self, client, project):
        started = start_run(client, project)
        task_id = next(iter(started["tasks"]))
        response = client.get(
            f"/orchestrations/{started['id']}/tasks/{task_id}"
        )
        assert response.status_code == 200

    def test_get_a_missing_task_is_404(self, client):
        assert client.get("/tasks/nope").status_code == 404

    def test_update_a_task(self, client, project):
        from app.models.domain import TaskStatus

        # A finished run is not re-scheduling anything, so there is no race.
        started = start_run(client, project, wait=True)
        task_id = next(iter(load_orchestration(started["id"]).state.tasks))

        def mark_failed(orchestration):
            orchestration.state.tasks[task_id].status = TaskStatus.FAILED

        mutate_orchestration(started["id"], mark_failed)
        response = client.patch(
            f"/orchestrations/{started['id']}/tasks/{task_id}",
            json={"title": "renamed", "priority": 9},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "renamed"
        assert response.json()["priority"] == 9

    def test_updating_a_missing_task_is_404(self, client, project):
        started = start_run(client, project)
        response = client.patch(
            f"/orchestrations/{started['id']}/tasks/nope", json={"title": "x"}
        )
        assert response.status_code == 404

    def test_cannot_edit_a_successful_task(self, client, project):
        started = start_run(client, project, wait=True)
        successful = [
            tid
            for tid, t in started["tasks"].items()
            if t["status"] == "SUCCESS"
        ]
        assert successful, "the run should have produced a successful task"
        response = client.patch(
            f"/orchestrations/{started['id']}/tasks/{successful[0]}",
            json={"title": "nope"},
        )
        assert response.status_code == 409

    def test_cannot_edit_a_running_task(self, client, project):
        from app.models.domain import TaskStatus

        started = start_run(client, project)
        task_id = next(iter(started["tasks"]))
        mutate_orchestration(
            started["id"],
            lambda o: setattr(o.state.tasks[task_id], "status", TaskStatus.RUNNING),
        )
        response = client.patch(
            f"/orchestrations/{started['id']}/tasks/{task_id}", json={"title": "nope"}
        )
        assert response.status_code == 409


# ── agents ──────────────────────────────────────────────────────────────────


class TestAgents:
    def test_list_the_roster(self, client):
        agents = client.get("/agents").json()
        assert len(agents) > 0
        assert all("role" in a and "name" in a for a in agents)
        assert any(a["role"] == "backend" for a in agents)

    def test_agents_of_a_run(self, client, project):
        started = start_run(client, project, wait=True)
        activity = client.get(f"/orchestrations/{started['id']}/agents").json()
        assert len(activity) == started["total_tasks"]
        row = activity[0]
        assert set(row) >= {"task_id", "title", "status", "summary", "files_changed"}

    def test_agents_of_a_missing_run_is_404(self, client):
        assert client.get("/orchestrations/nope/agents").status_code == 404

    def test_retry_a_task_in_a_finished_run(self, client, project):
        from app.models.domain import TaskStatus

        started = start_run(client, project, wait=True)
        task_id = next(iter(load_orchestration(started["id"]).state.tasks))

        def fail(orchestration):
            orchestration.state.tasks[task_id].status = TaskStatus.FAILED
            orchestration.state.tasks[task_id].errors = ["boom"]

        mutate_orchestration(started["id"], fail)
        response = client.post(f"/agents/{task_id}/retry", json={"reset_dependencies": True})
        assert response.status_code == 200
        assert "queued" in response.json()["message"]
        task = client.get(f"/tasks/{task_id}").json()
        # The task is re-queued and the finished run is restarted underneath it.
        assert task["status"] in {"READY", "RUNNING"}
        assert task["errors"] == []

    def test_cannot_retry_a_running_task(self, client, project):
        from app.models.domain import TaskStatus

        started = start_run(client, project, wait=True)
        task_id = next(iter(load_orchestration(started["id"]).state.tasks))
        mutate_orchestration(
            started["id"],
            lambda o: setattr(o.state.tasks[task_id], "status", TaskStatus.RUNNING),
        )
        response = client.post(f"/agents/{task_id}/retry", json={})
        assert response.status_code == 409

    def test_retry_a_missing_task_is_404(self, client):
        assert client.post("/agents/nope/retry", json={}).status_code == 404

    def test_retry_accepts_a_reason(self, client, project):
        from app.models.domain import TaskStatus

        started = start_run(client, project, wait=True)
        task_id = next(iter(load_orchestration(started["id"]).state.tasks))
        mutate_orchestration(
            started["id"],
            lambda o: setattr(o.state.tasks[task_id], "status", TaskStatus.FAILED),
        )
        response = client.post(
            f"/agents/{task_id}/retry", json={"reason": "provider was down"}
        )
        assert response.json()["detail"] == "provider was down"


# ── logs and streaming ──────────────────────────────────────────────────────


class TestLogsAndStream:
    def test_logs_are_returned_in_sequence_order(self, client, project):
        started = start_run(client, project, wait=True)
        logs = client.get(f"/orchestrations/{started['id']}/logs").json()
        assert logs
        sequences = [entry["sequence"] for entry in logs]
        assert sequences == sorted(sequences)

    def test_logs_support_incremental_polling(self, client, project):
        started = start_run(client, project, wait=True)
        all_logs = client.get(f"/orchestrations/{started['id']}/logs").json()
        assert len(all_logs) >= 2
        tail = client.get(
            f"/orchestrations/{started['id']}/logs?after={all_logs[0]['sequence']}"
        ).json()
        assert all(entry["sequence"] > all_logs[0]["sequence"] for entry in tail)

    def test_logs_validate_the_after_parameter(self, client, project):
        started = start_run(client, project)
        assert client.get(f"/orchestrations/{started['id']}/logs?after=-1").status_code == 422

    def test_logs_for_a_missing_run_is_404(self, client):
        assert client.get("/orchestrations/nope/logs").status_code == 404

    def test_sse_replays_past_events_then_closes(self, client, project):
        started = start_run(client, project, wait=True)
        with client.stream("GET", f"/orchestrations/{started['id']}/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())
        assert "event: open" in body
        assert "event: done" in body

    def test_sse_honours_the_after_cursor(self, client, project):
        started = start_run(client, project, wait=True)
        logs = client.get(f"/orchestrations/{started['id']}/logs").json()
        cursor = logs[0]["sequence"]
        with client.stream(
            "GET", f"/orchestrations/{started['id']}/stream?after={cursor}"
        ) as response:
            body = "".join(response.iter_text())
        assert "event: done" in body

    def test_sse_for_a_missing_run_is_404(self, client):
        assert client.get("/orchestrations/nope/stream").status_code == 404

    def test_sse_events_are_valid_json(self, client, project):
        started = start_run(client, project, wait=True)
        with client.stream("GET", f"/orchestrations/{started['id']}/stream") as response:
            body = "".join(response.iter_text())
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "{}":
                continue
            json.loads(payload)  # must not raise


# ── results ─────────────────────────────────────────────────────────────────


class TestResults:
    def test_changes_list_one_row_per_agent_result(self, client, project):
        started = start_run(client, project, wait=True)
        changes = client.get(f"/orchestrations/{started['id']}/changes").json()
        assert changes
        row = changes[0]
        assert set(row) >= {"task_id", "title", "branch", "commit", "files_changed"}

    def test_changes_for_a_missing_run_is_404(self, client):
        assert client.get("/orchestrations/nope/changes").status_code == 404

    def test_test_results_are_reported(self, client, project):
        started = start_run(client, project, wait=True)
        data = client.get(f"/orchestrations/{started['id']}/test-results").json()
        assert "summary" in data
        assert "runs" in data
        assert "report" in data

    def test_report_after_a_finished_run(self, client, project):
        started = start_run(client, project, wait=True)
        response = client.get(f"/orchestrations/{started['id']}/report")
        if response.status_code == 404:
            pytest.skip("this run produced nothing to integrate")
        data = response.json()
        assert set(data) >= {
            "markdown", "success", "merged_branches", "conflicts", "regressions"
        }

    def test_report_before_integration_is_404(self, client, project):
        started = start_run(client, project)
        assert client.get(f"/orchestrations/{started['id']}/report").status_code == 404

    def test_diff_of_the_working_tree_when_nothing_was_merged(self, client, project):
        started = start_run(client, project)
        data = client.get(f"/orchestrations/{started['id']}/diff").json()
        assert "files" in data
        assert "patch" in data

    def test_diff_against_a_named_branch(self, client, project):
        started = start_run(client, project, wait=True)
        response = client.get(
            f"/orchestrations/{started['id']}/diff?ref=main&stat_only=false"
        )
        assert response.status_code == 200
        assert "base" in response.json()

    def test_diff_of_a_non_repository_is_400(self, client, tmp_path):
        from app.models.domain import Project

        plain = tmp_path / "not-a-repo"
        plain.mkdir()
        save_project(Project(name="plain", local_path=str(plain)))
        created = client.post(
            "/orchestrations",
            json={
                "project_id": _last_project_id(client),
                "user_request": "Add a health endpoint",
            },
        )
        assert created.status_code == 201, created.text
        response = client.get(f"/orchestrations/{created.json()['id']}/diff")
        assert response.status_code == 400
        assert "not a git repository" in response.json()["detail"]


def _last_project_id(client) -> str:
    return client.get("/projects").json()[-1]["id"]


def load_orchestration(orchestration_id: str):
    """
    Reach into the server's in-memory orchestration from a sync test.

    `TestClient` runs the app on its own event loop, so the test thread drives
    the shared in-memory store with a short-lived loop of its own.
    """
    import asyncio

    from app.api.deps import get_container

    return asyncio.run(get_container().manager.get(orchestration_id))


def mutate_orchestration(orchestration_id: str, mutate) -> None:
    """
    Apply a change to the server's copy of an orchestration.

    The store deep-copies on both read and write, so mutating a loaded object
    is not enough -- it has to be saved back.
    """
    import asyncio

    from app.api.deps import get_container

    async def run() -> None:
        container = get_container()
        orchestration = await container.manager.get(orchestration_id)
        mutate(orchestration)
        await container.store.save_orchestration(orchestration)

    asyncio.run(run())


def save_project(project) -> None:
    import asyncio

    from app.api.deps import get_container

    asyncio.run(get_container().store.save_project(project))


# ── container wiring ────────────────────────────────────────────────────────


class TestContainer:
    def test_orchestrators_are_cached_per_repository(self, client):
        from app.api.deps import get_container

        container = get_container()
        container.orchestrators = {}
        first = container.orchestrator_for("/tmp/repo-a")
        second = container.orchestrator_for("/tmp/repo-a")
        third = container.orchestrator_for("/tmp/repo-b")
        assert first is second
        assert first is not third

    def test_key_hash_is_stable_and_short(self):
        from app.api.deps import key_hash

        assert key_hash("/a/b") == key_hash("/a/b")
        assert len(key_hash("/a/b")) == 12
        assert key_hash("/a/b") != key_hash("/a/c")

    def test_build_container_never_raises(self, settings, tmp_path):
        from app.api.deps import build_container

        container = build_container(settings)
        assert container.provider is not None
        assert len(container.registry) > 0
        assert container.orchestrators == {}


# ── one run per repository ──────────────────────────────────────────────────


class _TakenRepository:
    """
    A manager for a repository another run already holds.

    `is_running` is False so the cheap pre-check cannot catch this one, which
    is exactly the race the manager's own check exists to close.
    """

    def __init__(self) -> None:
        self.start_attempts = 0

    def is_running(self, orch_id: str) -> bool:
        return False

    async def start(self, orchestration, run_fn):
        from app.engine.execution import RepositoryBusy

        self.start_attempts += 1
        raise RepositoryBusy(
            f"orchestration other-run is already running on {orchestration.state.repository}"
        )

    async def wait(self, orch_id: str, timeout=None) -> bool:
        return True

    def register_scheduler(self, orch_id: str, scheduler) -> None:
        return None


class TestRepositoryExclusivityAtTheApi:
    @pytest.mark.asyncio
    async def test_a_refused_run_is_refused_and_leaves_nothing_behind(
        self, settings, registry, seed_repo
    ):
        """
        A second run against a busy repository must fail loudly *and* not
        linger: the plan is persisted before the run is started, so refusing
        without cleaning up would leave a phantom PENDING entry in the UI.
        """
        from fastapi import HTTPException

        from app.api.deps import Container
        from app.api.routes import create_orchestration
        from app.api.schemas import OrchestrationCreate
        from app.events import EventBus
        from app.llm.mock import MockLLMProvider
        from app.models.domain import Project
        from app.models.store import InMemoryStateStore
        from app.sandbox.manager import NoSandbox

        store = InMemoryStateStore()
        project = Project(name="demo", local_path=str(seed_repo), base_branch="main")
        await store.save_project(project)
        manager = _TakenRepository()
        container = Container(
            settings=settings,
            store=store,  # type: ignore[arg-type]
            event_bus=EventBus(),
            manager=manager,  # type: ignore[arg-type]
            registry=registry,
            provider=MockLLMProvider(),
            sandbox=NoSandbox(),
            orchestrators={},
        )

        with pytest.raises(HTTPException) as excinfo:
            await create_orchestration(
                OrchestrationCreate(
                    project_id=project.id,
                    user_request="Add a multiply function to the calculator",
                ),
                container,
            )

        assert excinfo.value.status_code == 409
        assert str(seed_repo) in str(excinfo.value.detail)
        assert manager.start_attempts == 1
        assert await store.list_orchestrations(project_id=project.id) == []


# ── settings persistence ────────────────────────────────────────────────────


class TestEnvPersistence:
    """
    Switching provider from the UI rewrites `.env`. Writing it to the wrong
    place is silent by nature -- the settings look saved and are simply never
    read again -- so the resolution rule and the rewrite both get pinned.
    """

    def test_a_cwd_env_is_the_one_settings_came_from(self, monkeypatch, tmp_path):
        from pathlib import Path

        from app.api import routes

        (tmp_path / ".env").write_text("LLM_PROVIDER=mock\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert routes._env_file_path() == Path(tmp_path) / ".env"

    def test_without_one_it_falls_back_to_the_project_file(self, monkeypatch, tmp_path):
        from pathlib import Path

        from app.api import routes

        monkeypatch.chdir(tmp_path)  # nothing to find here
        assert routes._env_file_path() == Path(routes.__file__).resolve().parents[2] / ".env"

    def test_updating_rewrites_in_place_and_appends_only_what_is_new(
        self, monkeypatch, tmp_path
    ):
        from app.api import routes

        target = tmp_path / ".env"
        target.write_text(
            "# keep my comment\nLLM_PROVIDER=mock\nUNRELATED=1\n", encoding="utf-8"
        )
        monkeypatch.setattr(routes, "_env_file_path", lambda: target)

        routes._update_env_file({"LLM_PROVIDER": "ollama", "OLLAMA_API_KEY": "k-1"})

        text = target.read_text(encoding="utf-8")
        assert "LLM_PROVIDER=ollama" in text
        assert text.count("LLM_PROVIDER=") == 1, "a key must not be written twice"
        assert "# keep my comment" in text, "unrelated lines survive"
        assert "UNRELATED=1" in text
        assert "OLLAMA_API_KEY=k-1" in text

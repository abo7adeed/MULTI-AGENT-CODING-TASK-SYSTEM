"""
Brain tests: understand the request, understand the repository, build a DAG.

The whole "planning" half of the system lives here, and it must be useful with
no model at all, so the majority of these tests run without a provider and the
LLM paths use a mock. The invariant that matters most is repeated throughout:
whatever the input, `decompose` must return a *valid* DAG.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.brain.analyzer import TaskAnalyzer, TaskSpecification
from app.brain.context import (
    ContextBudget,
    ContextManager,
    RepositorySnapshot,
    detect_conventions,
    scan_repository,
)
from app.brain.decomposer import BLUEPRINTS, TaskDecomposer
from app.brain.repo_analyzer import RepoAnalysis, RepositoryAnalyzer
from app.engine.dag import DAGEngine
from app.llm.mock import MockLLMProvider, ScriptedLLM
from app.models.domain import Task, TaskStatus, TaskType

REQUESTS = {
    "rag": "Build a RAG app with vector search and a chat API",
    "auth": "Add JWT authentication and login endpoints to the backend",
    "frontend": "Create a React dashboard with charts and a settings page",
    "fullstack": "Build a full stack todo app with auth, postgres and tests",
    "infra": "Containerise the app with Docker and add a CI/CD pipeline",
    "refactor": "Refactor the payment module, clean up tech debt and add tests",
    "docs": "Write a README and API documentation for the existing service",
    "vague": "make it better somehow",
}


# ── repository scanning ─────────────────────────────────────────────────────


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    root = tmp_path / "pyproj"
    (root / "src" / "app").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src" / "app" / "main.py").write_text(
        "from fastapi import FastAPI\nimport pytest\napp = FastAPI()\n", encoding="utf-8"
    )
    (root / "src" / "app" / "models.py").write_text(
        "from sqlalchemy.orm import DeclarativeBase\n", encoding="utf-8"
    )
    (root / "tests" / "test_main.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")
    return root


class TestScanRepository:
    def test_counts_files_and_languages(self, python_repo):
        snapshot = scan_repository(python_repo)
        assert snapshot.total_files > 0
        assert snapshot.languages["python"] == 3
        assert "src/app/main.py" in snapshot.files

    def test_skips_vendor_and_hidden_directories(self, tmp_path):
        root = tmp_path / "r"
        (root / "node_modules" / "x").mkdir(parents=True)
        (root / "__pycache__").mkdir()
        (root / ".hidden").mkdir()
        (root / "node_modules" / "x" / "a.js").write_text("x\n", encoding="utf-8")
        (root / "keep.py").write_text("x\n", encoding="utf-8")
        snapshot = scan_repository(root)
        assert snapshot.files == ["keep.py"]

    def test_skips_binary_extensions(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        (root / "logo.png").write_bytes(b"\x89PNG")
        (root / "lib.so").write_bytes(b"\x7fELF")
        (root / "app.py").write_text("x\n", encoding="utf-8")
        assert scan_repository(root).files == ["app.py"]

    def test_respects_max_files(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        for i in range(50):
            (root / f"f{i}.py").write_text("x\n", encoding="utf-8")
        snapshot = scan_repository(root, max_files=10)
        assert len(snapshot.files) <= 10
        assert snapshot.total_files <= 11

    def test_missing_directory_is_an_empty_snapshot(self, tmp_path):
        snapshot = scan_repository(tmp_path / "does-not-exist")
        assert snapshot.total_files == 0
        assert snapshot.files == []

    def test_classifies_tests_configs_and_entry_points(self, python_repo):
        snapshot = scan_repository(python_repo)
        assert "tests/test_main.py" in snapshot.test_files
        assert "pyproject.toml" in snapshot.config_files
        assert "src/app/main.py" in snapshot.entry_points

    def test_conventions_are_derived_not_guessed(self, python_repo):
        conventions = detect_conventions(scan_repository(python_repo))
        joined = " ".join(conventions)
        assert "Python project" in joined
        assert "pyproject.toml" in joined

    def test_conventions_fall_back_when_nothing_is_detected(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        assert detect_conventions(scan_repository(root)) == [
            "No established conventions detected"
        ]

    def test_snapshot_to_dict_is_serialisable(self, python_repo):
        data = scan_repository(python_repo).to_dict()
        json.dumps(data)  # must not raise
        assert data["total_files"] > 0


# ── repository analysis ─────────────────────────────────────────────────────


class TestRepositoryAnalyzer:
    @pytest.mark.asyncio
    async def test_detects_python_frameworks_from_manifests_and_sources(
        self, python_repo
    ):
        analysis = await RepositoryAnalyzer().analyze(python_repo, use_llm=False)
        assert {"fastapi", "sqlalchemy", "pytest"} <= set(analysis.frameworks)
        assert analysis.primary_language == "python"

    @pytest.mark.asyncio
    async def test_detects_containerisation_and_ci(self, python_repo):
        analysis = await RepositoryAnalyzer().analyze(python_repo, use_llm=False)
        assert analysis.dockerised is True
        assert analysis.has_ci is True
        assert "pip" in analysis.package_managers

    @pytest.mark.asyncio
    async def test_test_frameworks_are_a_subset_of_frameworks(self, python_repo):
        analysis = await RepositoryAnalyzer().analyze(python_repo, use_llm=False)
        assert set(analysis.test_frameworks) <= set(analysis.frameworks)
        assert "pytest" in analysis.test_frameworks

    @pytest.mark.asyncio
    async def test_detects_a_react_frontend(self, tmp_path):
        root = tmp_path / "web"
        root.mkdir()
        (root / "package.json").write_text(
            json.dumps({"dependencies": {"react": "18", "next": "14", "tailwind": "3"}}),
            encoding="utf-8",
        )
        (root / "src").mkdir()
        (root / "src" / "App.tsx").write_text(
            'import React from "react";\nexport default () => <div/>;\n', encoding="utf-8"
        )
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert {"react", "next", "tailwind"} <= set(analysis.frameworks)
        assert "npm" in analysis.package_managers
        assert analysis.languages.get("typescript") == 1

    @pytest.mark.asyncio
    async def test_go_and_rust_manifests(self, tmp_path):
        root = tmp_path / "polyglot"
        root.mkdir()
        (root / "go.mod").write_text("module x\n", encoding="utf-8")
        (root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert {"go", "cargo"} <= set(analysis.package_managers)

    @pytest.mark.asyncio
    async def test_empty_directory_is_reported_as_greenfield(self, tmp_path):
        root = tmp_path / "blank"
        root.mkdir()
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert analysis.total_files == 0
        assert "greenfield" in analysis.summary
        assert analysis.primary_language == "unknown"

    @pytest.mark.asyncio
    async def test_capability_hints_track_the_detected_stack(self, python_repo):
        analysis = await RepositoryAnalyzer().analyze(python_repo, use_llm=False)
        hints = analysis.capability_hints()
        assert TaskType.BACKEND in hints
        assert TaskType.DATABASE in hints
        assert TaskType.DEVOPS in hints
        assert TaskType.TESTING in hints
        assert TaskType.AI_ML not in hints  # no AI framework in this fixture

    @pytest.mark.asyncio
    async def test_capability_hints_include_ai_when_an_ai_framework_is_present(
        self, tmp_path
    ):
        root = tmp_path / "ai"
        root.mkdir()
        (root / "chain.py").write_text("from langgraph.graph import StateGraph\n", encoding="utf-8")
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert TaskType.AI_ML in analysis.capability_hints()

    @pytest.mark.asyncio
    async def test_capability_hints_for_a_bare_repository(self, tmp_path):
        root = tmp_path / "blank"
        root.mkdir()
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert analysis.capability_hints() == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "marker",
        [
            ".github/workflows",
            ".gitlab",
            ".circleci",
            ".azure-pipelines",
        ],
    )
    async def test_ci_is_detected_even_though_dot_dirs_are_not_walked(
        self, tmp_path, marker
    ):
        root = tmp_path / "ci"
        ci_dir = root / marker
        ci_dir.mkdir(parents=True)
        (ci_dir / "pipeline.yml").write_text("on: push\n", encoding="utf-8")
        (root / "app.py").write_text("x = 1\n", encoding="utf-8")
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert analysis.has_ci is True
        assert TaskType.DEVOPS in analysis.capability_hints()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name", ["azure-pipelines.yml", ".travis.yml", "Jenkinsfile", ".drone.yml"]
    )
    async def test_ci_is_detected_from_a_root_level_pipeline_file(self, tmp_path, name):
        root = tmp_path / "ci2"
        root.mkdir()
        (root / name).write_text("pipeline: true\n", encoding="utf-8")
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert analysis.has_ci is True

    @pytest.mark.asyncio
    async def test_an_empty_workflows_directory_is_not_ci(self, tmp_path):
        root = tmp_path / "ci3"
        (root / ".github" / "workflows").mkdir(parents=True)
        (root / "app.py").write_text("x = 1\n", encoding="utf-8")
        analysis = await RepositoryAnalyzer().analyze(root, use_llm=False)
        assert analysis.has_ci is False

    @pytest.mark.asyncio
    async def test_no_ci_is_reported_for_a_plain_repository(self, python_repo, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "app.py").write_text("x = 1\n", encoding="utf-8")
        analysis = await RepositoryAnalyzer().analyze(plain, use_llm=False)
        assert analysis.has_ci is False
        assert analysis.dockerised is False

    @pytest.mark.asyncio
    async def test_directory_map_counts_top_level_entries(self, python_repo):
        analysis = await RepositoryAnalyzer().analyze(python_repo, use_llm=False)
        assert "src" in analysis.directory_map
        assert analysis.directory_map["src"] == 2

    @pytest.mark.asyncio
    async def test_to_dict_is_serialisable_and_truncates(self, python_repo):
        analysis = await RepositoryAnalyzer().analyze(python_repo, use_llm=False)
        json.dumps(analysis.to_dict())
        assert len(analysis.to_dict()["test_files"]) <= 30

    @pytest.mark.asyncio
    async def test_llm_insight_is_parsed_from_json(self, python_repo):
        provider = ScriptedLLM(
            responses=[
                '{"architecture": "a layered API", "extension_points": ["src/app"], '
                '"risks": ["none"], "relevant_modules": ["src/app/main.py"]}'
            ]
        )
        analysis = await RepositoryAnalyzer(llm=provider).analyze(python_repo)
        insight = json.loads(analysis.llm_insight)
        assert insight["architecture"] == "a layered API"
        assert insight["relevant_modules"] == ["src/app/main.py"]

    @pytest.mark.asyncio
    async def test_llm_failure_leaves_the_deterministic_analysis_intact(
        self, python_repo
    ):
        provider = MockLLMProvider()
        provider.fail_times = 99
        provider.max_retries = 0
        analysis = await RepositoryAnalyzer(llm=provider).analyze(python_repo)
        assert analysis.llm_insight == ""
        assert analysis.frameworks  # still useful without a model

    @pytest.mark.asyncio
    async def test_non_json_llm_reply_is_kept_as_text(self, python_repo):
        analysis = await RepositoryAnalyzer(
            llm=MockLLMProvider(response="It looks like a small FastAPI service.")
        ).analyze(python_repo)
        assert analysis.llm_insight == "It looks like a small FastAPI service."

    @pytest.mark.asyncio
    async def test_use_llm_false_makes_no_call(self, python_repo):
        provider = MockLLMProvider()
        await RepositoryAnalyzer(llm=provider).analyze(python_repo, use_llm=False)
        assert provider.prompts == []

    def test_repo_analysis_defaults(self):
        analysis = RepoAnalysis()
        assert analysis.primary_language == "unknown"
        assert analysis.to_dict()["total_files"] == 0


# ── task analysis ───────────────────────────────────────────────────────────


class TestTaskAnalyzer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(REQUESTS))
    async def test_never_raises_and_always_returns_a_spec(self, name):
        spec = await TaskAnalyzer().analyze(REQUESTS[name])
        assert isinstance(spec, TaskSpecification)
        assert spec.raw_request == REQUESTS[name]
        assert spec.task_types

    @pytest.mark.asyncio
    async def test_analysis_is_always_the_first_task_type(self):
        for request in REQUESTS.values():
            spec = await TaskAnalyzer().analyze(request)
            assert spec.task_types[0] is TaskType.ANALYSIS

    @pytest.mark.asyncio
    async def test_detects_backend_and_authentication(self):
        spec = await TaskAnalyzer().analyze(
            "Add JWT authentication and login endpoints to the backend"
        )
        assert TaskType.SECURITY in spec.task_types
        assert TaskType.BACKEND in spec.task_types
        assert "authentication" in spec.deliverables
        assert "backend API" in spec.deliverables

    @pytest.mark.asyncio
    async def test_detects_ai_ml_for_a_rag_request(self):
        spec = await TaskAnalyzer().analyze(
            "Build a RAG app with vector search and a chat API"
        )
        assert TaskType.AI_ML in spec.task_types

    @pytest.mark.asyncio
    async def test_detects_frontend(self):
        spec = await TaskAnalyzer().analyze("Create a React dashboard with charts")
        assert TaskType.FRONTEND in spec.task_types
        assert "user interface" in spec.deliverables

    @pytest.mark.asyncio
    async def test_detects_devops(self):
        spec = await TaskAnalyzer().analyze("Containerise with Docker and add CI/CD")
        assert TaskType.DEVOPS in spec.task_types
        assert "container setup" in spec.deliverables
        assert "CI/CD pipeline" in spec.deliverables

    @pytest.mark.asyncio
    async def test_detects_refactor(self):
        spec = await TaskAnalyzer().analyze("Refactor the payment module and clean up")
        assert TaskType.REFACTOR in spec.task_types

    @pytest.mark.asyncio
    async def test_vague_request_gets_fallback_types(self):
        spec = await TaskAnalyzer().analyze("make it better somehow")
        assert spec.task_types == [TaskType.ANALYSIS, TaskType.PLANNING, TaskType.GENERIC]
        assert spec.ambiguities

    @pytest.mark.asyncio
    async def test_ambiguity_markers_are_reported(self):
        spec = await TaskAnalyzer().analyze("Add caching, or something like that")
        assert any("or something" in a for a in spec.ambiguities)

    @pytest.mark.asyncio
    async def test_a_precise_request_has_no_vagueness_warnings(self):
        spec = await TaskAnalyzer().analyze("Add a GET /health endpoint")
        assert spec.ambiguities == []

    @pytest.mark.asyncio
    async def test_greenfield_when_the_repository_is_empty(self):
        spec = await TaskAnalyzer().analyze("Build a new service", {"total_files": 0})
        assert spec.is_greenfield is True
        assert spec.intent.startswith("Greenfield build")

    @pytest.mark.asyncio
    async def test_not_greenfield_when_files_exist(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint", {"total_files": 42})
        assert spec.is_greenfield is False
        assert "Modify the existing codebase" in spec.intent

    @pytest.mark.asyncio
    async def test_keywords_exclude_stopwords_and_are_ranked(self):
        spec = await TaskAnalyzer().analyze("Build a payment payment payment service")
        assert spec.keywords[0] == "payment"
        assert "build" not in spec.keywords
        assert "the" not in spec.keywords

    @pytest.mark.asyncio
    async def test_title_is_the_first_clause_and_bounded(self):
        assert (await TaskAnalyzer().analyze("Add auth. Then add billing")).title == "Add auth"
        long = await TaskAnalyzer().analyze("x" * 300)
        assert len(long.title) <= 90

    @pytest.mark.asyncio
    async def test_title_strips_a_leading_please(self):
        spec = await TaskAnalyzer().analyze("Please add a login endpoint")
        assert not spec.title.lower().startswith("please")

    @pytest.mark.asyncio
    async def test_acceptance_criteria_depend_on_deliverables(self):
        with_tests = await TaskAnalyzer().analyze("Add an endpoint and write tests")
        assert "Automated tests exist for the new behaviour and pass" in (
            with_tests.acceptance_criteria
        )
        without = await TaskAnalyzer().analyze("Add an endpoint")
        assert "Existing test suite still passes" in without.acceptance_criteria

    @pytest.mark.asyncio
    async def test_acceptance_criteria_always_mentions_secrets(self):
        spec = await TaskAnalyzer().analyze("Add a feature")
        assert "No secrets or credentials are committed" in spec.acceptance_criteria

    @pytest.mark.asyncio
    async def test_constraints_include_package_manager_and_python_version(self):
        spec = await TaskAnalyzer().analyze(
            "Add an endpoint",
            {"package_managers": ["pnpm"], "python_version": "3.11"},
        )
        assert any("pnpm" in c for c in spec.constraints)
        assert any("3.11" in c for c in spec.constraints)

    @pytest.mark.asyncio
    async def test_complexity_grows_with_scope(self):
        simple = await TaskAnalyzer().analyze("Fix a typo in the docs")
        big = await TaskAnalyzer().analyze(
            "Build a full stack app with auth, postgres, RAG, Docker, CI and docs"
        )
        order = {"simple": 0, "moderate": 1, "complex": 2, "very large": 3}
        assert order[simple.complexity] < order[big.complexity]

    @pytest.mark.asyncio
    async def test_primary_type_falls_back_to_generic(self):
        assert TaskSpecification().primary_type is TaskType.GENERIC

    @pytest.mark.asyncio
    async def test_llm_enrichment_merges_criteria_without_duplicates(self):
        provider = ScriptedLLM(
            responses=[
                '{"summary": "s", "steps": ["one"], "risks": ["r"], '
                '"acceptance_criteria": ["Add an endpoint", "Handle errors"]}'
            ]
        )
        spec = await TaskAnalyzer(llm=provider).analyze("Add an endpoint")
        assert "Handle errors" in spec.acceptance_criteria
        assert spec.acceptance_criteria.count("Add an endpoint") == 1
        assert spec.source == "llm+heuristic"
        assert json.loads(spec.llm_plan)["steps"] == ["one"]

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_the_heuristic_spec(self):
        provider = MockLLMProvider()
        provider.fail_times = 99
        provider.max_retries = 0
        spec = await TaskAnalyzer(llm=provider).analyze("Add an endpoint")
        assert spec.llm_plan == ""
        assert spec.source == "heuristic"
        assert spec.task_types

    @pytest.mark.asyncio
    async def test_use_llm_false_makes_no_call(self):
        provider = MockLLMProvider()
        await TaskAnalyzer(llm=provider).analyze("Add an endpoint", use_llm=False)
        assert provider.prompts == []

    def test_to_dict_is_serialisable(self):
        json.dumps(
            TaskSpecification(
                raw_request="x", title="x", task_types=[TaskType.BACKEND]
            ).to_dict()
        )


# ── context building ────────────────────────────────────────────────────────


class TestContextManager:
    @pytest.fixture
    def snapshot(self, python_repo) -> RepositorySnapshot:
        return scan_repository(python_repo)

    def test_builds_a_context_for_a_task(self, snapshot, python_repo, tmp_path):
        task = Task(title="Add a health endpoint", type=TaskType.BACKEND)
        ctx = ContextManager(snapshot).build(
            task=task, workspace=str(tmp_path), repository=str(python_repo)
        )
        assert ctx.task is task
        assert ctx.workspace == str(tmp_path)
        assert ctx.repository == str(python_repo)
        assert ctx.relevant_files

    def test_relevant_files_are_capped_by_the_budget(self, snapshot, tmp_path):
        task = Task(title="anything", type=TaskType.GENERIC)
        ctx = ContextManager(snapshot, ContextBudget(max_files=2)).build(
            task=task, workspace=str(tmp_path)
        )
        assert len(ctx.relevant_files) <= 2

    def test_content_budget_is_enforced(self, snapshot, python_repo, tmp_path):
        task = Task(title="main models", type=TaskType.BACKEND)
        ctx = ContextManager(snapshot, ContextBudget(max_chars=200)).build(
            task=task, workspace=str(tmp_path), repository=str(python_repo)
        )
        assert sum(len(c) for c in ctx.file_contents.values()) <= 200 + 64

    def test_individual_files_are_truncated(self, snapshot, python_repo, tmp_path):
        (python_repo / "src" / "app" / "huge.py").write_text(
            "\n".join(f"line {i}" for i in range(5000)), encoding="utf-8"
        )
        snap = scan_repository(python_repo)
        ctx = ContextManager(snap, ContextBudget(max_file_chars=100)).build(
            task=Task(title="huge", type=TaskType.BACKEND),
            workspace=str(tmp_path),
            repository=str(python_repo),
        )
        for content in ctx.file_contents.values():
            assert len(content) <= 101

    def test_binary_and_missing_files_are_skipped(self, snapshot, python_repo, tmp_path):
        ctx = ContextManager(snapshot).build(
            task=Task(title="x", type=TaskType.BACKEND),
            workspace=str(tmp_path),
            repository=str(python_repo),
        )
        assert "logo.png" not in ctx.file_contents

    def test_dependency_summaries_are_included(self, snapshot, tmp_path):
        from app.models.domain import AgentResult, new_id

        dep_id = new_id()
        dep_result = AgentResult(
            task_id=dep_id,
            status=TaskStatus.SUCCESS,
            summary="built the schema",
            files_changed=["db.py"],
            output={"interfaces": ["UserRepository"]},
        )
        task = Task(title="build api", type=TaskType.BACKEND, dependencies=[dep_id])
        ctx = ContextManager(snapshot).build(
            task=task,
            workspace=str(tmp_path),
            dependency_results={dep_id: dep_result},
            agent_results={dep_id: dep_result},
        )
        assert ctx.dependencies[0]["summary"] == "built the schema"
        assert ctx.dependencies[0]["files_changed"] == ["db.py"]
        assert "UserRepository" in ctx.interfaces
        assert "built the schema" in ctx.upstream_summaries

    def test_unknown_dependency_does_not_crash(self, snapshot, tmp_path):
        task = Task(title="x", type=TaskType.BACKEND, dependencies=["ghost"])
        ctx = ContextManager(snapshot).build(task=task, workspace=str(tmp_path))
        assert ctx.dependencies[0]["summary"] == ""

    def test_retry_context_is_carried_through(self, snapshot, tmp_path):
        ctx = ContextManager(snapshot).build(
            task=Task(title="x", type=TaskType.BACKEND),
            workspace=str(tmp_path),
            attempt=3,
            previous_error="SyntaxError on line 2",
            previous_output={"ignored_file_blocks": 1},
            test_failures=["test_health failed"],
        )
        assert ctx.attempt == 3
        assert "SyntaxError" in ctx.previous_error
        assert ctx.test_failures == ["test_health failed"]
        assert ctx.previous_output["ignored_file_blocks"] == 1

    def test_role_scope_and_guidance_are_applied(self, snapshot, tmp_path):
        from app.agents.roster import BY_KEY

        ctx = ContextManager(snapshot).build(
            task=Task(title="x", type=TaskType.BACKEND),
            workspace=str(tmp_path),
            role_def=BY_KEY["backend"],
        )
        assert ctx.allowed_prefixes == list(BY_KEY["backend"].allowed_prefixes)
        assert ctx.instructions

    def test_empty_snapshot_still_builds(self, tmp_path):
        ctx = ContextManager(RepositorySnapshot()).build(
            task=Task(title="x", type=TaskType.GENERIC), workspace=str(tmp_path)
        )
        assert ctx.relevant_files == []
        assert ctx.conventions == []


# ── decomposition ───────────────────────────────────────────────────────────


def assert_valid(dag) -> None:
    DAGEngine(dag).validate()
    for task in dag.tasks.values():
        for dep in task.dependencies:
            assert dep in dag.tasks
            assert dep != task.id


class TestDecomposer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(REQUESTS))
    async def test_every_request_produces_a_valid_dag(self, name):
        spec = await TaskAnalyzer().analyze(REQUESTS[name])
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        assert_valid(dag)
        assert len(dag.tasks) >= 1

    @pytest.mark.asyncio
    async def test_blueprints_are_keyed_and_unique(self):
        keys = [b.key for b in BLUEPRINTS]
        assert len(keys) == len(set(keys))

    @pytest.mark.asyncio
    async def test_no_task_depends_on_a_dropped_blueprint(self):
        """The fixed point must not leave an orphan whose parent was pruned."""
        spec = await TaskAnalyzer().analyze("Add a login endpoint")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        keys = {b.key for b in BLUEPRINTS}
        assert_valid(dag)
        for task in dag.tasks.values():
            for dep_id in task.dependencies:
                parent = dag.tasks[dep_id].input_context.get("blueprint")
                assert parent is None or parent in keys

    @pytest.mark.asyncio
    async def test_the_spine_is_always_present(self):
        for request in REQUESTS.values():
            spec = await TaskAnalyzer().analyze(request)
            dag = await TaskDecomposer().decompose(spec, use_llm=False)
            keys = {t.input_context.get("blueprint") for t in dag.tasks.values()}
            assert "analysis" in keys

    @pytest.mark.asyncio
    async def test_a_backend_request_has_an_implementation_node(self):
        spec = await TaskAnalyzer().analyze("Add a REST API endpoint for users")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        types = {t.type for t in dag.tasks.values()}
        assert TaskType.BACKEND in types

    @pytest.mark.asyncio
    async def test_a_frontend_request_has_frontend_work(self):
        spec = await TaskAnalyzer().analyze("Create a React dashboard")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        assert TaskType.FRONTEND in {t.type for t in dag.tasks.values()}

    @pytest.mark.asyncio
    async def test_tests_depend_on_implementation_not_the_other_way_round(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint and write tests")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        by_key = {t.input_context.get("blueprint"): t for t in dag.tasks.values()}
        if "tests" in by_key:
            assert by_key["tests"].dependencies
            for dep in by_key["tests"].dependencies:
                assert by_key["tests"].id not in dag.tasks[dep].dependencies

    @pytest.mark.asyncio
    async def test_max_tasks_is_respected(self):
        spec = await TaskAnalyzer().analyze(
            "Build a full stack app with auth, postgres, RAG, Docker, CI, docs and tests"
        )
        dag = await TaskDecomposer(max_tasks=6).decompose(spec, use_llm=False)
        assert len(dag.tasks) <= 6
        assert_valid(dag)

    @pytest.mark.asyncio
    async def test_spec_fields_reach_the_task_context(self):
        spec = await TaskAnalyzer().analyze("Add a login endpoint")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        for task in dag.tasks.values():
            assert task.input_context["original_request"] == spec.raw_request
            assert "acceptance_criteria" in task.input_context

    @pytest.mark.asyncio
    async def test_titles_carry_the_request_subject(self):
        spec = await TaskAnalyzer().analyze("Build a payment gateway")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        assert any("payment" in t.title.lower() for t in dag.tasks.values())

    @pytest.mark.asyncio
    async def test_titles_are_ascii_only(self):
        spec = await TaskAnalyzer().analyze("Build a payment gateway with OAuth")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        for task in dag.tasks.values():
            task.title.encode("ascii")  # would raise on cp1252-hostile output

    @pytest.mark.asyncio
    async def test_apply_refinement_retitles_by_the_id_shown_to_the_model(self):
        """The id the model sees is the 8-char prefix, not the full uuid."""
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        target = next(iter(dag.tasks.values()))
        prefix = target.id[:8]
        refined = TaskDecomposer()._apply_refinement(
            dag, [{"id": prefix, "title": "Renamed by the model", "type": "security"}]
        )
        assert refined.tasks[target.id].title == "Renamed by the model"
        assert refined.tasks[target.id].type is TaskType.SECURITY

    @pytest.mark.asyncio
    async def test_apply_refinement_rewires_dependencies(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint and write tests")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        ids = [t.id[:8] for t in dag.tasks.values()]
        refined = TaskDecomposer()._apply_refinement(
            dag, [{"id": ids[0], "depends_on": [ids[-1]]}]
        )
        assert refined.tasks[[t for t in dag.tasks if t[:8] == ids[0]][0]].dependencies == [
            [t for t in dag.tasks if t[:8] == ids[-1]][0]
        ]

    @pytest.mark.asyncio
    async def test_apply_refinement_refuses_a_self_dependency(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        target = next(iter(dag.tasks.values()))
        before = list(target.dependencies)
        TaskDecomposer()._apply_refinement(dag, [{"id": target.id[:8], "depends_on": [target.id[:8]]}])
        assert target.dependencies == before
        assert_valid(dag)

    @pytest.mark.asyncio
    async def test_llm_can_add_a_new_node(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        base = await TaskDecomposer().decompose(spec, use_llm=False)
        provider = ScriptedLLM(
            responses=[
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "brand-new",
                                "title": "Extra hardening pass",
                                "type": "security",
                                "description": "check the new endpoint",
                            }
                        ]
                    }
                )
            ]
        )
        dag = await TaskDecomposer(llm=provider).decompose(spec, use_llm=True)
        assert_valid(dag)
        titles = [t.title for t in dag.tasks.values()]
        assert "Extra hardening pass" in titles
        assert len(dag.tasks) == len(base.tasks) + 1
        added = next(t for t in dag.tasks.values() if t.title == "Extra hardening pass")
        assert added.type is TaskType.SECURITY
        assert added.input_context["source"] == "llm_refinement"

    @pytest.mark.asyncio
    async def test_llm_cannot_introduce_a_cycle(self):
        spec = await TaskAnalyzer().analyze("Build a full stack app with auth and tests")
        base = await TaskDecomposer().decompose(spec, use_llm=False)
        ids = [t.id[:8] for t in base.tasks.values()]
        if len(ids) < 2:
            pytest.skip("need at least two tasks to build a cycle")
        provider = ScriptedLLM(
            responses=[
                json.dumps(
                    {
                        "tasks": [
                            {"id": ids[0], "depends_on": [ids[-1]]},
                            {"id": ids[-1], "depends_on": [ids[0]]},
                        ]
                    }
                )
            ]
        )
        dag = await TaskDecomposer(llm=provider).decompose(spec, use_llm=False)
        assert_valid(dag)

    @pytest.mark.asyncio
    async def test_llm_cannot_reference_a_task_that_does_not_exist(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        provider = ScriptedLLM(
            responses=[
                json.dumps({"tasks": [{"id": "aaaaaaaa", "depends_on": ["bbbbbbbb"]}]})
            ]
        )
        dag = await TaskDecomposer(llm=provider).decompose(spec, use_llm=True)
        assert_valid(dag)

    @pytest.mark.asyncio
    async def test_llm_replies_with_nonsense_are_ignored(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        baseline = await TaskDecomposer().decompose(spec, use_llm=False)
        for reply in ("I cannot help", "", "{}", '{"tasks": "not a list"}', "[]"):
            provider = MockLLMProvider(response=reply)
            dag = await TaskDecomposer(llm=provider).decompose(spec, use_llm=True)
            assert provider.prompts, "the refinement call should have been made"
            assert_valid(dag)
            assert len(dag.tasks) == len(baseline.tasks)

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_the_blueprint_dag(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        baseline = await TaskDecomposer().decompose(spec, use_llm=False)
        provider = MockLLMProvider()
        provider.fail_times = 99
        provider.max_retries = 0
        dag = await TaskDecomposer(llm=provider).decompose(spec, use_llm=True)
        assert len(dag.tasks) == len(baseline.tasks)
        assert_valid(dag)

    @pytest.mark.asyncio
    async def test_use_llm_false_makes_no_call(self):
        provider = MockLLMProvider()
        spec = await TaskAnalyzer().analyze("Add an endpoint")
        await TaskDecomposer(llm=provider).decompose(spec, use_llm=False)
        assert provider.prompts == []

    @pytest.mark.asyncio
    async def test_waves_execute_implementation_before_verification(self):
        spec = await TaskAnalyzer().analyze("Add an endpoint and write tests")
        dag = await TaskDecomposer().decompose(spec, use_llm=False)
        waves = DAGEngine(dag).parallel_waves()
        flat = [tid for wave in waves for tid in wave]
        by_key = {t.id: t.input_context.get("blueprint") for t in dag.tasks.values()}
        if "tests" in by_key.values():
            tests_index = flat.index(
                next(tid for tid in flat if by_key[tid] == "tests")
            )
            impl_indices = [
                flat.index(tid)
                for tid in flat
                if by_key[tid] not in {"tests", "integration", "review"}
            ]
            assert tests_index > min(impl_indices)

    @pytest.mark.asyncio
    async def test_decomposition_is_deterministic_in_shape(self):
        """Task ids are random uuids, so compare the structure, not the ids."""

        def shape(dag):
            by_key = {
                t.input_context.get("blueprint") or t.id[:8]: {
                    "title": t.title,
                    "type": t.type.value,
                    "priority": t.priority,
                    "deps": sorted(
                        dag.tasks[d].input_context.get("blueprint") or d[:8]
                        for d in t.dependencies
                    ),
                }
                for t in dag.tasks.values()
            }
            return by_key

        spec = await TaskAnalyzer().analyze(
            "Build a full stack todo app with auth and tests"
        )
        first = await TaskDecomposer().decompose(spec, use_llm=False)
        second = await TaskDecomposer().decompose(spec, use_llm=False)
        assert shape(first) == shape(second)

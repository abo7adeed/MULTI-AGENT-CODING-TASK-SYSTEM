"""Agent layer: roster, registry, selector, patch protocol, mock and LLM agents."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.agents import (
    AgentRegistry,
    AgentSelector,
    LLMCoderAgent,
    MockAgent,
    apply_patch,
    build_mock_roster,
    is_protected,
    parse_patch,
)
from app.agents.base import AgentContext
from app.agents.patch import PatchError, resolve_within
from app.agents.roster import BY_KEY, ROSTER
from app.llm.mock import MockLLMProvider, ScriptedLLM
from app.models.domain import Task, TaskStatus, TaskType
from app.models.agent_context import AgentContext as AgentContextModel


# ── roster ───────────────────────────────────────────────────────────────────

class TestRoster:
    def test_all_ten_required_roles_present(self):
        required = {
            "planner",
            "repository_analyst",
            "backend",
            "frontend",
            "database",
            "ai_ml",
            "testing",
            "devops",
            "debugging",
            "code_review",
        }
        assert required <= set(BY_KEY)

    def test_every_role_is_well_formed(self):
        for role in ROSTER:
            assert role.key and role.name and role.description and role.guidance
            assert role.capabilities
            assert 1 <= role.priority <= 9
            assert isinstance(role.task_types, tuple) and role.task_types

    def test_roles_can_be_materialised_as_agents(self):
        for role in ROSTER:
            model = role.to_agent()
            assert model.role == role.key
            assert model.name == role.name

    def test_read_only_roles(self):
        assert BY_KEY["planner"].writes_files is False
        assert BY_KEY["code_review"].writes_files is False
        assert BY_KEY["backend"].writes_files is True


# ── registry ─────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = MockAgent(role="backend")
        registry.register("backend", agent)
        assert registry.get("backend") is agent

    @pytest.mark.parametrize(
        "alias", ["backend_agent", "BACKEND", "backend", "Backend_Agent"]
    )
    def test_aliases_resolve(self, alias):
        registry = AgentRegistry()
        agent = MockAgent(role="backend")
        registry.register("backend", agent)
        assert registry.get(alias) is agent

    def test_unknown_role_raises_with_helpful_message(self):
        registry = AgentRegistry()
        registry.register("backend", MockAgent(role="backend"))
        with pytest.raises(KeyError) as exc:
            registry.get("nope")
        assert "backend" in str(exc.value)

    def test_find_returns_none_instead_of_raising(self):
        assert AgentRegistry().find("nope") is None

    def test_register_all_accepts_a_mapping(self):
        registry = AgentRegistry()
        registry.register_all(build_mock_roster())
        assert len(registry) == len(ROSTER)
        assert registry.has("backend")

    def test_register_all_accepts_an_iterable(self):
        registry = AgentRegistry()
        registry.register_all([MockAgent(role="backend"), MockAgent(role="planner")])
        assert set(registry.roles) == {"backend", "planner"}

    def test_describe_uses_roster_metadata(self):
        registry = AgentRegistry()
        registry.register_all(build_mock_roster())
        described = {a.role: a for a in registry.describe()}
        assert described["backend"].name == "Backend Coding Agent"
        assert described["backend"].metadata["implementation"] == "MockAgent"

    def test_unregister(self):
        registry = AgentRegistry()
        registry.register("backend", MockAgent(role="backend"))
        assert registry.unregister("backend") is True
        assert registry.has("backend") is False


# ── selector ─────────────────────────────────────────────────────────────────

class TestSelector:
    @pytest.mark.parametrize(
        "task_type,expected",
        [
            (TaskType.BACKEND, "backend"),
            (TaskType.FRONTEND, "frontend"),
            (TaskType.DATABASE, "database"),
            (TaskType.AI_ML, "ai_ml"),
            (TaskType.TESTING, "testing"),
            (TaskType.DEVOPS, "devops"),
            (TaskType.PLANNING, "planner"),
            (TaskType.ANALYSIS, "repository_analyst"),
            (TaskType.SECURITY, "security"),
            (TaskType.REVIEW, "code_review"),
            (TaskType.INTEGRATION, "integration"),
            (TaskType.DOCUMENTATION, "documentation"),
        ],
    )
    def test_type_routing(self, task_type, expected):
        selection = AgentSelector().select(Task(title="x", type=task_type))
        assert selection.role == expected
        assert selection.reason

    def test_explicit_assignment_wins(self):
        task = Task(title="x", type=TaskType.BACKEND, assigned_agent="frontend_agent")
        assert AgentSelector().select(task).role == "frontend"

    def test_keyword_fallback(self):
        task = Task(title="Fix the failing traceback in checkout", type="mystery")
        assert AgentSelector().select(task).role == "debugging"

    def test_unmatched_falls_back_to_generic(self):
        task = Task(title="zzz qqq", type="mystery", description="")
        selection = AgentSelector().select(task)
        assert selection.role == "generic"
        assert "no rule matched" in selection.reason

    def test_deterministic(self):
        selector = AgentSelector()
        task = Task(title="Add docker image", type="mystery")
        assert [selector.select(task).role for _ in range(5)] == ["devops"] * 5

    def test_roles_for_type(self):
        assert "backend" in AgentSelector().roles_for_type(TaskType.BACKEND)


# ── patch protocol ───────────────────────────────────────────────────────────

PATCH_TEXT = """
Here is what I did.

<file path="app/api/routes.py">
from fastapi import APIRouter

router = APIRouter()
</file>

<file path="tests/test_routes.py">
def test_router():
    assert True
</file>

<delete path="app/legacy.py"/>

<note>Added the router and a smoke test.</note>
"""


class TestPatchParsing:
    def test_extracts_everything(self):
        patch = parse_patch(PATCH_TEXT)
        assert [w.path for w in patch.writes] == ["app/api/routes.py", "tests/test_routes.py"]
        assert patch.deletes == ["app/legacy.py"]
        assert patch.notes == ["Added the router and a smoke test."]

    def test_strips_model_indentation(self):
        patch = parse_patch(PATCH_TEXT)
        assert patch.writes[0].content.startswith("from fastapi")
        assert not patch.writes[0].content.startswith("    from fastapi")

    def test_single_quoted_paths(self):
        assert parse_patch("<file path='a/b.py'>x</file>").writes[0].path == "a/b.py"

    def test_dot_slash_normalised(self):
        assert parse_patch("<file path='./a/b.py'>x</file>").writes[0].path == "a/b.py"

    def test_no_envelope_is_empty_not_an_error(self):
        patch = parse_patch("I could not do that, sorry.")
        assert patch.is_empty
        assert patch.writes == []

    def test_delete_that_is_also_written_is_dropped(self):
        patch = parse_patch('<file path="a.py">x</file><delete path="a.py"/>')
        assert patch.deletes == []


class TestPatchSecurity:
    @pytest.mark.parametrize(
        "path",
        [
            "../../escape.txt",
            "../../../etc/passwd",
            "a/../../b.txt",
            "..",
            ".env",
            ".env.production",
            ".git/config",
            "nested/.git/hooks/pre-commit",
            "id_rsa",
            "certs/server.pem",
            "keys/private.key",
            "~/.ssh/authorized_keys",
            "/etc/hosts",
            "C:/Windows/system32/drivers/etc/hosts",
        ],
    )
    def test_dangerous_paths_are_rejected(self, path, tmp_path):
        result = apply_patch(parse_patch(f'<file path="{path}">x</file>'), tmp_path)
        assert result.written == [], f"{path} escaped the sandbox"
        assert result.rejected

    def test_rejection_reason_is_actionable(self, tmp_path):
        result = apply_patch(parse_patch('<file path="../../x">y</file>'), tmp_path)
        assert "escapes" in result.rejected[0]["reason"] or "protected" in result.rejected[0]["reason"]

    def test_nothing_lands_outside_the_workspace(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        apply_patch(parse_patch('<file path="../sibling.txt">x</file>'), workspace)
        assert not (tmp_path / "sibling.txt").exists()

    def test_protected_paths_are_flagged(self):
        assert is_protected(".env")
        assert is_protected(".git/config")
        assert is_protected("../x")
        assert is_protected("/abs")
        assert not is_protected("app/main.py")

    def test_resolve_within_raises_outside_apply(self, tmp_path):
        with pytest.raises(PatchError):
            resolve_within(tmp_path, "../nope")

    def test_gitignore_is_not_protected_but_env_is(self, tmp_path):
        ok = apply_patch(parse_patch('<file path=".gitignore">x</file>'), tmp_path)
        assert ok.written == [".gitignore"]
        bad = apply_patch(parse_patch('<file path=".env">SECRET=1</file>'), tmp_path)
        assert bad.written == []

    def test_a_path_the_filesystem_refuses_is_a_rejection_not_a_crash(self, tmp_path):
        """
        An over-long or illegal filename is the agent's mistake to correct, the
        same as an out-of-scope path. Letting the OSError escape killed the
        agent outright instead of feeding the repair loop a reason.
        """
        too_long = "a" * 300 + ".py"
        result = apply_patch(parse_patch(f'<file path="{too_long}">x</file>'), tmp_path)

        assert result.written == []
        assert result.rejected, "an unwritable path must be reported, not raised"
        assert "cannot write" in result.rejected[0]["reason"]

    def test_a_rejected_write_leaves_no_partial_file_behind(self, tmp_path):
        apply_patch(parse_patch(f'<file path="{"a" * 300}.py">x</file>'), tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestPatchApplication:
    def test_writes_and_deletes(self, tmp_path):
        (tmp_path / "gone.py").write_text("old", encoding="utf-8")
        result = apply_patch(parse_patch(PATCH_TEXT), tmp_path)
        assert result.written == ["app/api/routes.py", "tests/test_routes.py"]
        assert result.deleted == ["app/legacy.py"]
        assert (tmp_path / "app" / "api" / "routes.py").exists()
        assert not (tmp_path / "app" / "legacy.py").exists()

    def test_creates_nested_directories(self, tmp_path):
        apply_patch(parse_patch('<file path="a/b/c/d.py">x</file>'), tmp_path)
        assert (tmp_path / "a" / "b" / "c" / "d.py").exists()

    def test_scope_restriction(self, tmp_path):
        result = apply_patch(
            parse_patch('<file path="frontend/App.tsx">x</file>'), tmp_path,
            allowed_prefixes=["app/"],
        )
        assert result.written == []
        assert "scope" in result.rejected[0]["reason"]

    def test_scope_allows_permitted_prefix(self, tmp_path):
        result = apply_patch(
            parse_patch('<file path="app/main.py">x</file>'), tmp_path,
            allowed_prefixes=["app/"],
        )
        assert result.written == ["app/main.py"]

    def test_dry_run_writes_nothing(self, tmp_path):
        result = apply_patch(parse_patch('<file path="a.py">x</file>'), tmp_path, dry_run=True)
        assert result.written == ["a.py"]
        assert not (tmp_path / "a.py").exists()

    def test_overwrite_is_reported_as_modified(self, tmp_path):
        (tmp_path / "a.py").write_text("old", encoding="utf-8")
        apply_patch(parse_patch('<file path="a.py">new</file>'), tmp_path)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new\n"


# ── mock agents ──────────────────────────────────────────────────────────────

def make_context(tmp_path, task: Task, **kwargs) -> AgentContext:
    """
    A context whose workspace looks like a real checkout.

    The LLM agent refuses an empty workspace before spending a model call, so
    the fixture has to contain at least the base files.
    """
    tmp_path = Path(tmp_path)
    (tmp_path / "README.md").write_text("# base\n", encoding="utf-8")
    return AgentContext(
        task=task,
        workspace=str(tmp_path),
        repository=str(tmp_path),
        **kwargs,
    )


class TestMockAgent:
    @pytest.mark.asyncio
    async def test_writes_real_files(self, tmp_path):
        task = Task(title="Implement widget", type=TaskType.BACKEND)
        result = await MockAgent(role="backend").execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.SUCCESS
        assert result.files_changed
        for rel in result.files_changed:
            assert (tmp_path / rel).exists()

    @pytest.mark.asyncio
    async def test_failure_mode(self, tmp_path):
        task = Task(title="x", type=TaskType.BACKEND)
        result = await MockAgent(role="backend", succeed=False).execute(
            task, make_context(tmp_path, task)
        )
        assert result.status is TaskStatus.FAILED
        assert result.errors

    @pytest.mark.asyncio
    async def test_fail_task_ids(self, tmp_path):
        task = Task(id="t1", title="x", type=TaskType.BACKEND)
        agent = MockAgent(role="backend", fail_task_ids=["t1"])
        assert (await agent.execute(task, make_context(tmp_path, task))).status is TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_transient_failures_recover(self, tmp_path):
        task = Task(id="t1", title="x", type=TaskType.BACKEND)
        agent = MockAgent(role="backend", fail_times=2)
        assert (await agent.execute(task, make_context(tmp_path, task))).status is TaskStatus.FAILED
        assert (await agent.execute(task, make_context(tmp_path, task))).status is TaskStatus.FAILED
        assert (await agent.execute(task, make_context(tmp_path, task))).status is TaskStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_read_only_role_writes_nothing(self, tmp_path):
        task = Task(title="Plan it", type=TaskType.PLANNING)
        result = await MockAgent(role="planner").execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.SUCCESS
        assert result.files_changed == []
        assert result.recommendations

    def test_roster_builder_covers_every_role(self):
        roster = build_mock_roster()
        assert set(roster) == set(BY_KEY)


# ── LLM coding agent ─────────────────────────────────────────────────────────

GOOD_REPLY = """
<file path="app/service.py">
def handle():
    return "ok"
</file>
<note>Added a handler.</note>
"""


class TestLLMCoderAgent:
    @pytest.mark.asyncio
    async def test_applies_a_well_formed_patch(self, tmp_path):
        agent = LLMCoderAgent(MockLLMProvider(response=GOOD_REPLY), role="backend")
        task = Task(title="Add a handler", type=TaskType.BACKEND)
        result = await agent.execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.SUCCESS
        assert (tmp_path / "app" / "service.py").exists()
        assert result.files_changed == ["app/service.py"]
        assert result.output["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_prose_only_reply_is_reported_as_no_change(self, tmp_path):
        agent = LLMCoderAgent(
            MockLLMProvider(response="I am afraid I cannot help with that."),
            role="backend",
        )
        task = Task(title="x", type=TaskType.BACKEND)
        result = await agent.execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.FAILED
        assert "no file changes" in result.summary

    @pytest.mark.asyncio
    async def test_out_of_scope_patch_is_repaired_then_fails_cleanly(self, tmp_path):
        agent = LLMCoderAgent(
            MockLLMProvider(response='<file path="frontend/App.tsx">x</file>'),
            role="backend",
            max_repair_rounds=1,
        )
        task = Task(title="x", type=TaskType.BACKEND)
        result = await agent.execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.FAILED
        assert not (tmp_path / "frontend").exists()

    @pytest.mark.asyncio
    async def test_path_traversal_from_the_model_is_blocked(self, tmp_path):
        agent = LLMCoderAgent(
            MockLLMProvider(response='<file path="../../pwned.txt">x</file>'),
            role="backend",
            max_repair_rounds=0,
        )
        task = Task(title="x", type=TaskType.BACKEND)
        result = await agent.execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.FAILED
        assert not (tmp_path.parent / "pwned.txt").exists()

    @pytest.mark.asyncio
    async def test_provider_failure_is_reported_not_raised(self, tmp_path):
        provider = MockLLMProvider()
        provider.fail_times = 99
        agent = LLMCoderAgent(provider, role="backend")
        task = Task(title="x", type=TaskType.BACKEND)
        result = await agent.execute(task, make_context(tmp_path, task))
        assert result.status is TaskStatus.FAILED
        assert result.errors

    @pytest.mark.asyncio
    async def test_empty_workspace_is_refused_before_spending_a_call(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()  # deliberately NOT seeded
        provider = ScriptedLLM(responses=["<file path='a.py'>x</file>"])
        agent = LLMCoderAgent(provider, role="backend")
        task = Task(title="x", type=TaskType.BACKEND)
        # Bypass make_context, which seeds the workspace.
        bare = AgentContext(task=task, workspace=str(empty), repository=str(empty))
        result = await agent.execute(task, bare)
        assert result.status is TaskStatus.FAILED
        assert "workspace is empty" in result.summary
        assert provider.prompts == []  # no model call was made

    @pytest.mark.asyncio
    async def test_read_only_agent_writes_nothing(self, tmp_path):
        agent = LLMCoderAgent(MockLLMProvider(response=GOOD_REPLY), role="planner")
        task = Task(title="Plan", type=TaskType.PLANNING)
        result = await agent.execute(task, make_context(tmp_path, task))
        assert result.files_changed == []
        assert not (tmp_path / "app").exists()

    @pytest.mark.asyncio
    async def test_prompt_contains_the_context_the_agent_needs(self, tmp_path):
        provider = MockLLMProvider(response="<note>done</note>")
        agent = LLMCoderAgent(provider, role="backend")
        task = Task(title="Add rate limiting", type=TaskType.BACKEND)
        context = make_context(
            tmp_path,
            task,
            original_request="Harden the API",
            upstream_summaries=["Repo analyst: FastAPI + SQLAlchemy"],
            test_failures=["test_limit: expected 429"],
            previous_error="previous attempt crashed",
        )
        await agent.execute(task, context)
        prompt = provider.prompts[0]
        assert "Add rate limiting" in prompt
        assert "Harden the API" in prompt
        assert "FastAPI + SQLAlchemy" in prompt
        assert "test_limit" in prompt
        assert "previous attempt crashed" in prompt


# ── agent context ────────────────────────────────────────────────────────────

class TestAgentContext:
    def test_prompt_dict_drops_empty_sections(self):
        task = Task(title="x", type=TaskType.BACKEND)
        context = AgentContext(task=task, workspace="/tmp")
        data = context.as_prompt_dict()
        assert data["task_title"] == "x"
        assert "test_failures" not in data
        assert "upstream_summaries" not in data

    def test_context_model_is_the_same_class(self):
        assert AgentContextModel is AgentContext

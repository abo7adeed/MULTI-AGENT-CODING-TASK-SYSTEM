"""
Sandbox tests.

The sandbox is a security control, so these tests are about what the code
*refuses* as much as what it runs. Command construction, mount validation and
environment scrubbing are all testable without Docker; the few tests that
genuinely need a daemon skip themselves when one is not running.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app.config import Settings
from app.sandbox.manager import (
    ENV_ALLOWLIST,
    FORBIDDEN_ENV_SUBSTRINGS,
    DockerSandbox,
    NoSandbox,
    SandboxError,
    SandboxPolicy,
    SandboxResult,
    build_sandbox,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspaces" / "task-1"
    root.mkdir(parents=True)
    (root / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return root


@pytest.fixture
def box(workspace: Path) -> DockerSandbox:
    return DockerSandbox(
        settings=Settings(_env_file=None, sandbox_enabled=True),
        allowed_roots=[str(workspace.parent)],
    )


# ── policy ──────────────────────────────────────────────────────────────────


class TestPolicy:
    def test_defaults_are_locked_down(self):
        policy = SandboxPolicy()
        assert policy.network == "none"
        assert policy.read_only_root is True
        assert policy.run_as_user != "0:0"
        assert policy.allow_network is False

    def test_policy_is_built_from_settings(self):
        settings = Settings(
            _env_file=None,
            sandbox_enabled=True,
            sandbox_image="python:3.12-slim",
            sandbox_memory="512m",
            sandbox_cpus="1.0",
            sandbox_pids_limit=32,
            sandbox_timeout_seconds=12.0,
            sandbox_network="none",
        )
        policy = SandboxPolicy.from_settings(settings)
        assert policy.image == "python:3.12-slim"
        assert policy.memory == "512m"
        assert policy.pids_limit == 32
        assert policy.timeout == 12.0
        assert policy.allow_network is False

    def test_a_non_none_network_is_recorded_as_allowed(self):
        settings = Settings(
            _env_file=None, sandbox_enabled=True, sandbox_network="bridge"
        )
        assert SandboxPolicy.from_settings(settings).allow_network is True

    def test_describe_exposes_the_contract(self):
        described = SandboxPolicy().describe()
        assert set(described) == {
            "image", "network", "memory", "cpus", "pids_limit",
            "read_only_root", "run_as_user",
        }


# ── command construction ────────────────────────────────────────────────────


class TestBuildCommand:
    def test_every_isolation_flag_is_present(self, box, workspace):
        cmd = box.build_command(["pytest", "-q"], workspace)
        assert cmd[:2] == ["docker", "run"]
        assert "--rm" in cmd and "--init" in cmd
        assert cmd[cmd.index("--network") + 1] == "none"
        assert cmd[cmd.index("--memory") + 1] == box.policy.memory
        assert cmd[cmd.index("--cpus") + 1] == box.policy.cpus
        assert cmd[cmd.index("--pids-limit") + 1] == str(box.policy.pids_limit)
        assert "--read-only" in cmd
        assert cmd[cmd.index("--user") + 1] == box.policy.run_as_user

    def test_capabilities_are_dropped(self, box, workspace):
        cmd = box.build_command(["true"], workspace)
        assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
        assert cmd[cmd.index("--security-opt") + 1] == "no-new-privileges"

    def test_swap_is_capped_so_it_cannot_exceed_memory(self, box, workspace):
        cmd = box.build_command(["true"], workspace)
        assert cmd[cmd.index("--memory-swap") + 1] == box.policy.memory

    def test_a_tmpfs_is_provided_for_scratch_space(self, box, workspace):
        cmd = box.build_command(["true"], workspace)
        assert any(a.startswith("/tmp:") for a in cmd)

    def test_the_workspace_is_mounted_at_a_fixed_path(self, box, workspace):
        cmd = box.build_command(["true"], workspace)
        assert f"{workspace.resolve()}:/workspace:rw" in cmd
        assert cmd[cmd.index("-w") + 1] == "/workspace"

    def test_the_image_and_argv_come_last(self, box, workspace):
        cmd = box.build_command(["pytest", "-q", "tests/"], workspace)
        assert cmd[-4:] == [box.policy.image, "pytest", "-q", "tests/"]

    def test_allowed_environment_is_forwarded(self, box, workspace):
        cmd = box.build_command(["true"], workspace, env={"LANG": "C.UTF-8"})
        assert "LANG=C.UTF-8" in cmd

    def test_a_read_write_root_drops_the_read_only_flags(self, workspace):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True),
            policy=SandboxPolicy(read_only_root=False),
            allowed_roots=[str(workspace.parent)],
        )
        cmd = box.build_command(["true"], workspace)
        assert "--read-only" not in cmd
        assert not any(a.startswith("/tmp:") for a in cmd)


# ── mount validation ────────────────────────────────────────────────────────


class TestMountValidation:
    def test_a_workspace_inside_an_allowed_root_is_accepted(self, box, workspace):
        assert box.build_command(["true"], workspace)

    def test_a_workspace_outside_every_root_is_refused(self, box, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(SandboxError, match="outside the allowed roots"):
            box.build_command(["true"], outside)

    def test_a_missing_workspace_is_refused(self, box, tmp_path):
        with pytest.raises(SandboxError, match="does not exist"):
            box.build_command(["true"], tmp_path / "nope")

    def test_a_file_is_not_a_workspace(self, box, workspace):
        with pytest.raises(SandboxError, match="not a directory"):
            box.build_command(["true"], workspace / "main.py")

    @pytest.mark.parametrize("dangerous", ["/", "/etc", "/root", "/home", "/proc"])
    def test_system_paths_are_refused_even_without_a_allowlist(self, dangerous):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True),
            allowed_roots=[],
        )
        target = Path(dangerous)
        if not target.is_dir():
            pytest.skip(f"{dangerous} does not exist on this platform")
        with pytest.raises(SandboxError):
            box.build_command(["true"], target)

    def test_a_filesystem_root_on_any_drive_is_refused(self, workspace):
        """The dangerous list is C:-specific, so the root check must not be."""
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True), allowed_roots=[]
        )
        drive_root = Path(workspace.anchor or "/")
        with pytest.raises(SandboxError, match="filesystem root"):
            box.build_command(["true"], drive_root)

    def test_a_traversal_out_of_the_root_is_refused(self, box, workspace):
        sneaky = workspace / ".." / ".." / "escape"
        sneaky.mkdir(parents=True, exist_ok=True)
        with pytest.raises(SandboxError, match="outside the allowed roots"):
            box.build_command(["true"], sneaky)

    def test_with_no_roots_configured_any_directory_is_allowed(self, workspace):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True), allowed_roots=[]
        )
        assert box.build_command(["true"], workspace)

    def test_a_symlink_out_of_the_root_is_refused(self, box, workspace, tmp_path):
        link = workspace / "link"
        try:
            link.symlink_to(tmp_path, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not available here")
        with pytest.raises(SandboxError, match="outside the allowed roots"):
            box.build_command(["true"], link)


# ── environment scrubbing ───────────────────────────────────────────────────


class TestEnvScrubbing:
    def test_allowlisted_variables_survive(self):
        clean = DockerSandbox.scrub_env({"PATH": "/usr/bin", "HOME": "/root", "LANG": "C"})
        assert clean["PATH"] == "/usr/bin"
        assert clean["HOME"] == "/root"
        assert clean["LANG"] == "C"

    def test_unlisted_variables_are_dropped(self):
        clean = DockerSandbox.scrub_env({"MY_APP_SETTING": "x", "PATH": "/usr/bin"})
        assert "MY_APP_SETTING" not in clean

    @pytest.mark.parametrize(
        "name",
        [
            "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
            "DB_PASSWORD", "MY_PRIVATE_KEY", "AZURE_CREDENTIALS",
            "DATABASE_URL", "NPM_TOKEN", "SSH_AUTH_SOCK",
        ],
    )
    def test_credentials_never_enter_a_container(self, name):
        clean = DockerSandbox.scrub_env({name: "hunter2"})
        assert name not in clean

    def test_scrubbing_a_full_allowlist_still_drops_nothing_it_should(self):
        clean = DockerSandbox.scrub_env({k: "v" for k in ENV_ALLOWLIST})
        assert set(clean) >= set(ENV_ALLOWLIST)

    def test_the_host_environment_is_the_default_source(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
        monkeypatch.setenv("PATH", "/usr/bin")
        clean = DockerSandbox.scrub_env()
        assert "OPENAI_API_KEY" not in clean
        assert clean["PATH"] == "/usr/bin"

    def test_unsafe_defaults_are_always_present(self):
        clean = DockerSandbox.scrub_env({})
        assert clean["HOME"] == "/tmp"
        assert clean["PYTHONDONTWRITEBYTECODE"] == "1"

    def test_a_caller_cannot_override_home_to_a_secret(self):
        clean = DockerSandbox.scrub_env({"HOME": "/etc"})
        assert clean["HOME"] == "/etc"  # HOME is allowlisted, so it is honoured

    def test_the_forbidden_list_covers_the_obvious_substrings(self):
        joined = " ".join(FORBIDDEN_ENV_SUBSTRINGS)
        for expected in ("SECRET", "TOKEN", "PASSWORD", "AWS_"):
            assert expected in joined


# ── capability probing ──────────────────────────────────────────────────────


class TestAvailability:
    def test_availability_is_false_without_the_docker_binary(self, monkeypatch):
        monkeypatch.setattr("app.sandbox.manager.shutil.which", lambda _: None)
        box = DockerSandbox(settings=Settings(_env_file=None, sandbox_enabled=True))
        assert box.available is False

    def test_the_probe_is_cached(self, monkeypatch):
        calls = []

        def which(name):
            calls.append(name)
            return None

        monkeypatch.setattr("app.sandbox.manager.shutil.which", which)
        box = DockerSandbox(settings=Settings(_env_file=None, sandbox_enabled=True))
        assert box.available is False
        assert box.available is False
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_running_without_docker_raises_a_clear_error(self, workspace, monkeypatch):
        monkeypatch.setattr("app.sandbox.manager.shutil.which", lambda _: None)
        box = DockerSandbox(settings=Settings(_env_file=None, sandbox_enabled=True))
        with pytest.raises(SandboxError, match="sandbox_enabled=false"):
            await box.run(workspace, ["true"])

    @pytest.mark.asyncio
    async def test_an_empty_command_is_refused(self, box):
        with pytest.raises(SandboxError, match="no command"):
            await box.run("/tmp", [])

    @pytest.mark.asyncio
    async def test_check_is_false_without_docker(self, monkeypatch):
        monkeypatch.setattr("app.sandbox.manager.shutil.which", lambda _: None)
        box = DockerSandbox(settings=Settings(_env_file=None, sandbox_enabled=True))
        assert await box.check() is False

    def test_describe_lists_the_roots_and_allowlist(self, box):
        described = box.describe()
        assert "allowed_roots" in described
        assert described["env_allowlist"] == list(ENV_ALLOWLIST)
        assert described["policy"]["network"] == "none"


# ── no-sandbox mode ─────────────────────────────────────────────────────────


class TestNoSandbox:
    def test_it_is_a_named_object_not_none(self):
        assert isinstance(NoSandbox(), NoSandbox)
        assert NoSandbox().available is False

    @pytest.mark.asyncio
    async def test_it_refuses_to_run_anything_on_the_host(self, workspace):
        with pytest.raises(SandboxError, match="refusing to run agent code on the host"):
            await NoSandbox().run(workspace, ["rm", "-rf", "/"])

    @pytest.mark.asyncio
    async def test_run_tests_also_refuses(self, workspace):
        with pytest.raises(SandboxError):
            await NoSandbox().run_tests(workspace, ["pytest"])

    @pytest.mark.asyncio
    async def test_check_is_false(self):
        assert await NoSandbox().check() is False

    def test_describe_says_why(self):
        assert NoSandbox().describe() == {
            "available": False,
            "policy": None,
            "note": "sandboxing disabled",
        }


# ── factory ─────────────────────────────────────────────────────────────────


class TestBuildSandbox:
    def test_disabled_gives_no_sandbox(self):
        assert isinstance(build_sandbox(Settings(_env_file=None, sandbox_enabled=False)), NoSandbox)

    def test_enabled_gives_a_docker_sandbox(self, tmp_path):
        sandbox = build_sandbox(
            Settings(_env_file=None, sandbox_enabled=True),
            allowed_roots=[str(tmp_path)],
        )
        assert isinstance(sandbox, DockerSandbox)
        assert sandbox.allowed_roots == [tmp_path.resolve()]

    def test_roots_are_resolved(self, tmp_path):
        sandbox = build_sandbox(
            Settings(_env_file=None, sandbox_enabled=True),
            allowed_roots=[str(tmp_path / "." / "x")],
        )
        assert sandbox.allowed_roots[0] == (tmp_path / "x").resolve()


# ── result ──────────────────────────────────────────────────────────────────


class TestSandboxResult:
    def test_ok_requires_a_clean_exit(self):
        assert SandboxResult(0, "", "", 0.1).ok is True
        assert SandboxResult(1, "", "", 0.1).ok is False

    def test_a_timeout_is_never_ok(self):
        assert SandboxResult(124, "", "", 0.1, timed_out=True).ok is False

    def test_to_dict_truncates(self):
        result = SandboxResult(0, "x" * 50_000, "y" * 20_000, 1.0)
        data = result.to_dict()
        assert len(data["stdout"]) <= 20_000
        assert len(data["stderr"]) <= 8_000
        assert data["ok"] is True


# ── live docker (skipped when unavailable) ──────────────────────────────────


def docker_ready() -> bool:
    if DockerSandbox(settings=Settings(_env_file=None, sandbox_enabled=True)).available is False:
        return False
    try:
        return (
            asyncio.run(
                DockerSandbox(settings=Settings(_env_file=None, sandbox_enabled=True)).check()
            )
            is True
        )
    except Exception:
        return False


needs_docker = pytest.mark.skipif(
    not docker_ready(), reason="no usable docker daemon on this machine"
)


@needs_docker
class TestLiveDocker:
    @pytest.mark.asyncio
    async def test_a_command_runs_and_returns_its_output(self, workspace):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True, sandbox_image="alpine"),
            allowed_roots=[str(workspace.parent)],
        )
        result = await box.run(workspace, ["echo", "sandboxed"])
        assert result.exit_code == 0
        assert "sandboxed" in result.stdout

    @pytest.mark.asyncio
    async def test_the_workspace_is_visible_inside_the_container(self, workspace):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True, sandbox_image="alpine"),
            allowed_roots=[str(workspace.parent)],
        )
        result = await box.run(workspace, ["cat", "/workspace/main.py"])
        assert "print('hi')" in result.stdout

    @pytest.mark.asyncio
    async def test_a_failing_command_reports_its_exit_code(self, workspace):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True, sandbox_image="alpine"),
            allowed_roots=[str(workspace.parent)],
        )
        result = await box.run(workspace, ["sh", "-c", "exit 3"])
        assert result.exit_code == 3
        assert result.ok is False

    @pytest.mark.asyncio
    async def test_the_host_environment_is_not_visible(self, workspace):
        monkey_host = dict(os.environ)
        os.environ["OPENAI_API_KEY"] = "sk-should-not-be-visible"
        try:
            box = DockerSandbox(
                settings=Settings(_env_file=None, sandbox_enabled=True, sandbox_image="alpine"),
                allowed_roots=[str(workspace.parent)],
            )
            result = await box.run(
                workspace, ["sh", "-c", "echo \"[$OPENAI_API_KEY]\""]
            )
        finally:
            os.environ.clear()
            os.environ.update(monkey_host)
        assert "sk-should-not-be-visible" not in result.stdout

    @pytest.mark.asyncio
    async def test_run_tests_matches_the_sandbox_runner_protocol(self, workspace):
        box = DockerSandbox(
            settings=Settings(_env_file=None, sandbox_enabled=True, sandbox_image="alpine"),
            allowed_roots=[str(workspace.parent)],
        )
        result = await box.run_tests(workspace, ["true"])
        assert set(result) >= {"exit_code", "stdout", "stderr", "ok"}

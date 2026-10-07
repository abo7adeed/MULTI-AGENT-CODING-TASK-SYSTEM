"""
DockerSandbox.

Agents execute code. That makes isolation a correctness property, not a
nicety, so every container is created with:

  * no network by default
  * an explicit memory / CPU / PID budget
  * a read-only root filesystem with a tmpfs for scratch space
  * no host environment variables (the env is built from an allowlist)
  * a wall-clock timeout with a hard kill
  * a non-root user

The one thing that is *not* optional is which host paths get mounted. Only the
task workspace is bind-mounted, and a path outside the configured root is
refused rather than clamped.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger("app.sandbox")

#: Never forwarded into a sandbox, regardless of what the host has set.
FORBIDDEN_ENV_SUBSTRINGS = (
    "SECRET", "TOKEN", "PASSWORD", "PASSWD", "APIKEY", "API_KEY",
    "PRIVATE_KEY", "CREDENTIAL", "AWS_", "GCP_", "AZURE_", "SSH_",
    "GITHUB_TOKEN", "GH_TOKEN", "NPM_TOKEN", "DATABASE_URL",
)

#: Only these may be passed through, even if the host sets them.
ENV_ALLOWLIST = (
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM", "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE", "NO_COLOR", "CI", "SYSTEMROOT", "COMSPEC",
    "PATHEXT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "APPDATA",
    "PROGRAMFILES", "HOMEDRIVE", "HOMEPATH", "NUMBER_OF_PROCESSORS",
)

_DANGEROUS_HOST_PATHS = (
    "/", "/etc", "/root", "/home", "/var/run", "/proc", "/sys", "/dev",
    "c:\\", "c:\\windows", "c:\\users",
)


class SandboxError(RuntimeError):
    """The sandbox refused or failed to run the request."""


@dataclass
class SandboxPolicy:
    """The isolation contract, resolved from settings."""

    image: str = "python:3.11-slim"
    network: str = "none"
    memory: str = "2g"
    cpus: str = "2.0"
    pids_limit: int = 256
    timeout: float = 600.0
    read_only_root: bool = True
    run_as_user: str = "1000:1000"
    allow_network: bool = False
    allowed_hosts: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: Settings) -> "SandboxPolicy":
        return cls(
            image=settings.sandbox_image,
            network="none" if not settings.sandbox_enabled else settings.sandbox_network,
            memory=settings.sandbox_memory,
            cpus=settings.sandbox_cpus,
            pids_limit=settings.sandbox_pids_limit,
            timeout=settings.sandbox_timeout_seconds,
            allow_network=bool(settings.sandbox_network and settings.sandbox_network != "none"),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "network": self.network,
            "memory": self.memory,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
            "read_only_root": self.read_only_root,
            "run_as_user": self.run_as_user,
        }


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout[-20_000:],
            "stderr": self.stderr[-8_000:],
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "ok": self.ok,
        }


class DockerSandbox:
    """
    Runs a command inside a disposable, resource-capped container.

        async with DockerSandbox(settings) as box:
            result = await box.run(workspace, ["pytest", "-q"])
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        policy: Optional[SandboxPolicy] = None,
        allowed_roots: Optional[Sequence[str]] = None,
    ):
        self.settings = settings or get_settings()
        self.policy = policy or SandboxPolicy.from_settings(self.settings)
        self.allowed_roots = [Path(p).resolve() for p in (allowed_roots or [])]
        self._checked_docker: Optional[bool] = None

    # ── capability ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether docker is usable right now. Cached after the first probe."""
        if self._checked_docker is None:
            self._checked_docker = shutil.which("docker") is not None
        return self._checked_docker

    # ── command construction ────────────────────────────────────────────────

    def build_command(
        self,
        argv: Sequence[str],
        workspace: str | Path,
        env: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """Assemble the full `docker run` argv with every isolation flag."""
        mount = self._validate_mount(workspace)
        container_workspace = "/workspace"
        cmd: list[str] = [
            "docker", "run", "--rm", "--init",
            "--network", self.policy.network,
            "--memory", self.policy.memory,
            "--memory-swap", self.policy.memory,   # no swap escape hatch
            "--cpus", self.policy.cpus,
            "--pids-limit", str(self.policy.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", self.policy.run_as_user,
            "-v", f"{mount}:{container_workspace}:rw",
            "-w", container_workspace,
        ]
        if self.policy.read_only_root:
            cmd += ["--read-only", "--tmpfs", "/tmp:rw,size=256m,mode=1777"]
        for key, value in self.scrub_env(env).items():
            cmd += ["-e", f"{key}={value}"]
        cmd.append(self.policy.image)
        cmd += list(argv)
        return cmd

    def _validate_mount(self, workspace: str | Path) -> str:
        """
        Refuse to mount anything outside the configured roots.

        This is the control that stops an agent from being handed the host's
        home directory by pointing its workspace at `C:\\Users`.
        """
        path = Path(workspace).expanduser().resolve()
        if not path.is_dir():
            raise SandboxError(f"workspace does not exist or is not a directory: {path}")

        # Any filesystem root, on any drive, plus the well-known system paths.
        if path.parent == path or path.anchor and str(path).rstrip("\\/") == path.anchor.rstrip("\\/"):
            raise SandboxError(f"refusing to mount a filesystem root: {path}")

        lowered = str(path).lower().rstrip("\\/")
        for dangerous in _DANGEROUS_HOST_PATHS:
            if lowered == dangerous.rstrip("\\/").lower():
                raise SandboxError(f"refusing to mount a system path: {path}")

        if self.allowed_roots:
            for root in self.allowed_roots:
                try:
                    path.relative_to(root)
                    break
                except ValueError:
                    continue
            else:
                raise SandboxError(
                    f"workspace {path} is outside the allowed roots "
                    f"{[str(r) for r in self.allowed_roots]}"
                )
        return str(path)

    @staticmethod
    def scrub_env(env: Optional[dict[str, str]] = None) -> dict[str, str]:
        """
        Build a container environment from an allowlist.

        Anything that smells like a credential is dropped even if the caller
        passes it explicitly -- the point is that a bug in a caller cannot
        leak a host secret into agent-controlled code.
        """
        source = env if env is not None else dict(os.environ)
        clean: dict[str, str] = {}
        for key, value in source.items():
            upper = key.upper()
            if upper not in {k.upper() for k in ENV_ALLOWLIST}:
                continue
            if any(bad in upper for bad in FORBIDDEN_ENV_SUBSTRINGS):
                continue
            clean[key] = value
        clean.setdefault("HOME", "/tmp")
        clean.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return clean

    # ── execution ───────────────────────────────────────────────────────────

    async def run(
        self,
        workspace: str | Path,
        argv: Sequence[str],
        timeout: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
    ) -> SandboxResult:
        """Run `argv` in the sandbox. Returns a result; never raises on failure."""
        if not argv:
            raise SandboxError("no command given")
        if not self.available:
            raise SandboxError(
                "docker is not available; cannot run in the sandbox. "
                "Set sandbox_enabled=false to run commands directly."
            )

        command = self.build_command(argv, workspace, env)
        limit = timeout or self.policy.timeout
        logger.info(
            "Sandbox run",
            extra={
                "argv": " ".join(str(a) for a in argv)[:300],
                "image": self.policy.image,
                "network": self.policy.network,
            },
        )

        import time

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=limit
            )
            return SandboxResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                duration_seconds=round(time.monotonic() - started, 3),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                exit_code=124,
                stdout="",
                stderr=f"sandboxed command exceeded {limit}s and was killed",
                duration_seconds=round(time.monotonic() - started, 3),
                timed_out=True,
            )

    async def run_tests(
        self, workspace: str | Path, argv: Sequence[str], timeout: Optional[float] = None
    ) -> dict[str, Any]:
        """Convenience wrapper matching the `sandbox_runner` protocol."""
        return (await self.run(workspace, argv, timeout)).to_dict()

    async def check(self) -> bool:
        """Probe the docker daemon."""
        if not self.available:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def describe(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "policy": self.policy.describe(),
            "allowed_roots": [str(p) for p in self.allowed_roots],
            "env_allowlist": list(ENV_ALLOWLIST),
        }


class NoSandbox:
    """
    Explicit no-op runner.

    Used when `sandbox_enabled` is false. It is a named object rather than
    `None` so the test runner and the API can report honestly which mode is in
    effect instead of silently running agent code on the host.
    """

    available = False

    async def run(
        self,
        workspace: str | Path,
        argv: Sequence[str],
        timeout: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
    ) -> SandboxResult:
        raise SandboxError(
            "sandboxing is disabled; refusing to run agent code on the host"
        )

    async def run_tests(
        self, workspace: str | Path, argv: Sequence[str], timeout: Optional[float] = None
    ) -> dict[str, Any]:
        # Parenthesised deliberately: `await self.run(...).to_dict()` binds the
        # attribute access to the coroutine and raises AttributeError instead
        # of surfacing the refusal.
        return (await self.run(workspace, argv, timeout)).to_dict()

    async def check(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        return {"available": False, "policy": None, "note": "sandboxing disabled"}


def build_sandbox(
    settings: Optional[Settings] = None, allowed_roots: Optional[Sequence[str]] = None
):
    """Return a DockerSandbox, or NoSandbox when isolation is switched off."""
    settings = settings or get_settings()
    if not settings.sandbox_enabled:
        return NoSandbox()
    return DockerSandbox(settings=settings, allowed_roots=allowed_roots)

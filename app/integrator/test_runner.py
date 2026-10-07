"""
TestRunner.

Runs the project's real test suite and reports what actually happened. The
previous version returned `True` unconditionally, which meant integration
"validation" validated nothing.

Command detection is deterministic: look at what the repository actually
contains, in priority order. An explicit override always wins.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.logging_config import get_logger
from app.models.domain import TestRunResult

logger = get_logger("app.integrator.tests")

#: Environment scrubbed from every test run: no secrets, no ambient config.
_CLEAN_ENV_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY",
                   "GITHUB_TOKEN", "GH_TOKEN", "API_KEY")

_PYTEST_PATTERNS = (
    re.compile(r"(?P<passed>\d+) passed"),
    re.compile(r"(?P<failed>\d+) failed"),
    re.compile(r"(?P<errors>\d+) error"),
    re.compile(r"(?P<skipped>\d+) skipped"),
)


@dataclass
class TestCommand:
    argv: list[str]
    kind: str          # pytest | npm | go | cargo | dotnet | custom
    label: str

    def __str__(self) -> str:
        return shlex.join(self.argv)


class TestRunner:
    """
    Detects and runs the project's test suite.

        runner = TestRunner()
        if runner.can_run(repo_path):
            result = await runner.run(repo_path)
    """

    def __init__(
        self,
        command: Optional[str] = None,
        timeout: float = 900.0,
        use_sandbox: bool = False,
        sandbox_runner=None,
    ):
        self.command = command
        self.timeout = timeout
        self.use_sandbox = use_sandbox
        self.sandbox_runner = sandbox_runner

    # ── detection ───────────────────────────────────────────────────────────

    def detect_command(self, repo_path: str | Path) -> Optional[TestCommand]:
        """Choose a test command from what the repository actually contains."""
        if self.command:
            return TestCommand(split_command(self.command), "custom", self.command)

        root = Path(repo_path)
        if not root.exists():
            return None

        if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (
            root / "tests").is_dir() or _has_python_tests(root):
            return TestCommand([sys_executable(), "-m", "pytest", "-q", "--no-header"], "pytest", "pytest")

        package_json = root / "package.json"
        if package_json.is_file():
            try:
                import json

                data = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
                scripts = data.get("scripts", {}) or {}
                for name in ("test", "test:ci"):
                    if name in scripts:
                        return TestCommand(
                            ["npm", "run", name, "--silent"], "npm", f"npm run {name}"
                        )
            except Exception:  # noqa: BLE001 - fall through to the next candidate
                pass
            if (root / "node_modules").is_dir():
                return TestCommand(["npm", "test"], "npm", "npm test")

        if (root / "go.mod").exists():
            return TestCommand(["go", "test", "./..."], "go", "go test ./...")
        if (root / "Cargo.toml").exists():
            return TestCommand(["cargo", "test"], "cargo", "cargo test")
        if (root / "Makefile").exists():
            try:
                if "test:" in (root / "Makefile").read_text(encoding="utf-8", errors="replace"):
                    return TestCommand(["make", "test"], "make", "make test")
            except OSError:
                pass
        return None

    def can_run(self, repo_path: str | Path) -> bool:
        return self.detect_command(repo_path) is not None

    # ── execution ───────────────────────────────────────────────────────────

    async def run(
        self,
        repo_path: str | Path,
        timeout: Optional[float] = None,
    ) -> TestRunResult:
        """Run the suite. Never raises: a broken suite is a result, not a crash."""
        command = self.detect_command(repo_path)
        if command is None:
            return TestRunResult(
                command="",
                passed=True,
                skipped=True,
                stdout="No recognisable test command in this repository; nothing to run.",
            )

        limit = timeout or self.timeout
        started = time.monotonic()
        logger.info("Running tests", extra={"command": str(command), "repo": str(repo_path)})

        if self.use_sandbox and self.sandbox_runner is not None:
            return await self._run_sandbox(repo_path, command, limit, started)

        try:
            proc = await asyncio.create_subprocess_exec(
                *command.argv,
                cwd=str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
            )
        except FileNotFoundError as exc:
            return TestRunResult(
                command=str(command),
                passed=False,
                exit_code=127,
                stderr=f"{command.argv[0]} not found: {exc}",
                duration_seconds=round(time.monotonic() - started, 3),
            )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=limit)
            code = proc.returncode or 0
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return TestRunResult(
                command=str(command),
                passed=False,
                exit_code=124,
                stderr=f"Test run exceeded {limit}s and was killed.",
                duration_seconds=round(time.monotonic() - started, 3),
            )

        result = TestRunResult(
            command=str(command),
            passed=code == 0,
            exit_code=code,
            stdout=out[-20_000:],
            stderr=err[-8_000:],
            duration_seconds=round(time.monotonic() - started, 3),
        )
        self._parse_counts(result)
        if not result.passed:
            logger.warning(
                "Tests failed",
                extra={
                    "command": str(command),
                    "exit_code": code,
                    "summary": result.summary,
                },
            )
        return result

    async def _run_sandbox(
        self, repo_path: str | Path, command: TestCommand, limit: float, started: float
    ) -> TestRunResult:
        try:
            outcome = await self.sandbox_runner.run(
                str(repo_path), command.argv, timeout=limit
            )
        except Exception as exc:  # noqa: BLE001
            return TestRunResult(
                command=str(command),
                passed=False,
                exit_code=1,
                stderr=f"sandbox failed: {exc}",
                duration_seconds=round(time.monotonic() - started, 3),
            )
        result = TestRunResult(
            command=str(command),
            passed=bool(outcome.get("exit_code") == 0),
            exit_code=int(outcome.get("exit_code", 1)),
            stdout=str(outcome.get("stdout", ""))[-20_000:],
            stderr=str(outcome.get("stderr", ""))[-8_000:],
            duration_seconds=round(time.monotonic() - started, 3),
        )
        self._parse_counts(result)
        return result

    @staticmethod
    def _env() -> dict[str, str]:
        """A minimal environment: no inherited secrets, no ambient colour."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(k.startswith(p) for p in _CLEAN_ENV_KEYS)
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["NO_COLOR"] = "1"
        env["CI"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    @staticmethod
    def _parse_counts(result: TestRunResult) -> None:
        """Extract pass/fail counts from pytest or generic output."""
        blob = f"{result.stdout}\n{result.stderr}"
        passed = failed = 0
        for pattern in _PYTEST_PATTERNS:
            match = pattern.search(blob)
            if not match:
                continue
            value = int(match.group(1))
            key = match.lastgroup or ""
            if key == "passed":
                passed = value
            elif key in ("failed", "errors"):
                failed += value
        if passed or failed:
            result.passed_count = passed
            result.failed_count = failed
            return
        # Fall back to counting result lines for non-pytest runners.
        result.passed_count = len(re.findall(r"^(ok|PASS|✓)", blob, re.M))
        result.failed_count = len(re.findall(r"^(FAIL|FAILED|✗|--- FAIL)", blob, re.M))

    # ── convenience ─────────────────────────────────────────────────────────

    async def run_with_retry(self, repo_path: str, attempts: int = 2) -> list[TestRunResult]:
        """Run, and re-run once. Useful for genuinely flaky suites."""
        results = [await self.run(repo_path)]
        for _ in range(max(0, attempts - 1)):
            if results[-1].passed:
                break
            results.append(await self.run(repo_path))
        return results


def sys_executable() -> str:
    import sys

    return sys.executable or "python"


def split_command(command: str) -> list[str]:
    """
    Split a command string into argv.

    `shlex.split` treats a backslash as an escape character, which silently
    turns `C:\\Python\\python.exe` into `C:Pythonpython.exe`. On Windows a
    backslash is a path separator, so escaping is switched off while quote
    handling is kept.
    """
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    return list(lexer)


def _has_python_tests(root: Path) -> bool:
    for pattern in ("test_*.py", "*_test.py"):
        if any(root.glob(pattern)) or any(root.rglob(pattern)):
            return True
    return False

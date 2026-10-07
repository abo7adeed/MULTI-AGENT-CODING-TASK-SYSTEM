"""
ContextManager.

Assembles the smallest context that lets an agent do its job. The whole point
is that agents do NOT receive the repository: they receive the handful of files
that matter, the interfaces they must honour, and a summary of what upstream
agents already decided.

Two hard rules:
  * relevance is scored deterministically (path, type, recency) -- never by an LLM
  * the result is capped by a character budget so cost stays bounded
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.models.agent_context import AgentContext
from app.logging_config import get_logger
from app.models.domain import AgentResult, Task, TaskStatus

logger = get_logger("app.brain.context")

#: Directories never read into agent context.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", ".next",
    "coverage", ".tox", ".idea", ".vscode", "workspaces", "state",
}

#: Binary-ish extensions never read into agent context.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".pdf", ".zip",
    ".gz", ".tar", ".whl", ".so", ".dll", ".dylib", ".exe", ".bin", ".lock",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".db", ".sqlite",
}

MAX_FILE_BYTES = 40_000
DEFAULT_BUDGET = 48_000  # characters


@dataclass
class ContextBudget:
    max_chars: int = DEFAULT_BUDGET
    max_files: int = 40
    max_file_chars: int = 8_000


@dataclass
class RepositorySnapshot:
    """What we learned by walking the repo once, reused for every task."""

    root: str = ""
    files: list[str] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    package_manifest: str = ""
    total_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "total_files": self.total_files,
            "languages": self.languages,
            "entry_points": self.entry_points[:20],
            "test_files": self.test_files[:20],
            "config_files": self.config_files[:20],
            "conventions": self.conventions,
            "package_manifest": self.package_manifest,
        }


_LANG_BY_SUFFIX = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".go": "go", ".rs": "rust",
    ".java": "java", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".sh": "shell", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".json": "json", ".toml": "toml",
}
_TEST_HINTS = ("test_", "_test.", ".test.", ".spec.", "/tests/", "/test/")
_CONFIG_NAMES = {
    "pyproject.toml", "requirements.txt", "setup.py", "package.json",
    "tsconfig.json", "dockerfile", "docker-compose.yml", "makefile",
    "pytest.ini", "tox.ini", "ruff.toml", ".eslintrc.json", "vite.config.ts",
    "alembic.ini", "go.mod", "cargo.toml",
}
_ENTRY_HINTS = ("main.py", "app.py", "server.py", "manage.py", "cli.py",
                "index.ts", "index.tsx", "main.ts", "app.tsx", "__main__.py")


def scan_repository(root: str | Path, max_files: int = 4000) -> RepositorySnapshot:
    """
    Walk the repository once and record what matters.

    Pure filesystem work, no LLM. This is what grounds every later decision.
    """
    root = Path(root)
    snapshot = RepositorySnapshot(root=str(root))
    if not root.exists():
        return snapshot

    counted = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, filename), root).replace("\\", "/")
            suffix = Path(filename).suffix.lower()
            if suffix in SKIP_SUFFIXES:
                continue
            counted += 1
            if counted > max_files:
                break
            snapshot.files.append(rel)
            lang = _LANG_BY_SUFFIX.get(suffix)
            if lang:
                snapshot.languages[lang] = snapshot.languages.get(lang, 0) + 1
            lowered = rel.lower()
            if any(h in lowered for h in _TEST_HINTS):
                snapshot.test_files.append(rel)
            elif filename.lower() in _CONFIG_NAMES:
                snapshot.config_files.append(rel)
            if filename.lower() in _ENTRY_HINTS:
                snapshot.entry_points.append(rel)
        if counted > max_files:
            break

    snapshot.total_files = counted
    snapshot.conventions = detect_conventions(snapshot)
    return snapshot


def detect_conventions(snapshot: RepositorySnapshot) -> list[str]:
    """Infer the house style so generated code matches the codebase."""
    conventions: list[str] = []
    langs = snapshot.languages
    if langs.get("python"):
        conventions.append("Python project: 4-space indent, type hints on public functions")
    if langs.get("typescript") or langs.get("javascript"):
        conventions.append("TypeScript/JS: strict typing, ES modules")
    if "pyproject.toml" in [Path(f).name for f in snapshot.config_files]:
        conventions.append("Configuration lives in pyproject.toml")
    if "package.json" in [Path(f).name for f in snapshot.config_files]:
        conventions.append("Node package: scripts in package.json")
    if snapshot.test_files:
        conventions.append(f"Tests live in {len(snapshot.test_files)} test file(s); follow their layout")
    if any(f.lower().startswith("dockerfile") for f in snapshot.config_files):
        conventions.append("Containerised: keep Dockerfiles in sync with dependencies")
    if not conventions:
        conventions.append("No established conventions detected")
    return conventions


class ContextManager:
    """
    Builds an `AgentContext` for one task.

        ctx = ContextManager(snapshot).build(task=..., workspace=..., state=...)
    """

    def __init__(
        self,
        snapshot: RepositorySnapshot | None = None,
        budget: ContextBudget | None = None,
    ):
        self.snapshot = snapshot or RepositorySnapshot()
        self.budget = budget or ContextBudget()

    # ── public API ──────────────────────────────────────────────────────────

    def build(
        self,
        task: Task,
        workspace: str,
        repository: str = "",
        original_request: str = "",
        role_def=None,
        dependency_results: dict[str, AgentResult] | None = None,
        agent_results: dict[str, AgentResult] | None = None,
        repository_analysis: dict[str, Any] | None = None,
        test_failures: Iterable[str] = (),
        previous_error: str = "",
        previous_output: dict[str, Any] | None = None,
        attempt: int = 1,
    ) -> AgentContext:
        dependency_results = dependency_results or {}
        agent_results = agent_results or {}

        dependencies = [
            {
                "task_id": dep,
                "title": self._title_of(dep, agent_results),
                "summary": self._summary_of(dep, agent_results),
                "files_changed": agent_results[dep].files_changed if dep in agent_results else [],
                "interfaces": agent_results[dep].output.get("interfaces", [])
                if dep in agent_results
                else [],
            }
            for dep in task.dependencies
        ]

        relevant = self.select_relevant_files(task)
        contents = self._read(relevant)

        context = AgentContext(
            task=task,
            workspace=workspace,
            repository=repository or self.snapshot.root,
            original_request=original_request,
            role=getattr(role_def, "key", ""),
            dependencies=dependencies,
            upstream_summaries=[
                d["summary"] for d in dependencies if d.get("summary")
            ],
            relevant_files=relevant,
            file_contents=contents,
            repository_analysis=repository_analysis or self.snapshot.to_dict(),
            interfaces=self._collect_interfaces(dependency_results),
            conventions=list(self.snapshot.conventions),
            test_failures=list(test_failures),
            attempt=attempt,
            previous_error=previous_error,
            previous_output=previous_output or {},
            allowed_prefixes=list(getattr(role_def, "allowed_prefixes", ()) or ()),
            instructions=getattr(role_def, "guidance", ""),
            test_requirements=self._test_requirements(task),
        )
        logger.debug(
            "Context built",
            extra={
                "task_title": task.title,
                "files": len(relevant),
                "chars": sum(len(c) for c in contents.values()),
            },
        )
        return context

    # ── file selection ──────────────────────────────────────────────────────

    def select_relevant_files(self, task: Task) -> list[str]:
        """
        Score every file against the task text and return the best few.

        Deterministic keyword overlap plus a boost for files in the role's
        allowed scope, so a backend agent sees `app/`, not `frontend/`.
        """
        if not self.snapshot.files:
            return []
        tokens = self._tokenize(f"{task.title} {task.description} {task.type.value}")
        allowed = _role_prefixes(task)

        scored: list[tuple[float, str]] = []
        for rel in self.snapshot.files:
            score = self._score(rel, tokens, allowed)
            if score > 0:
                scored.append((score, rel))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [rel for _, rel in scored[: self.budget.max_files]]

    def _score(self, rel: str, tokens: set[str], allowed: tuple[str, ...]) -> float:
        score = 0.0
        lowered = rel.lower()
        parts = set(re.split(r"[^a-z0-9]+", lowered)) - {""}

        overlap = tokens & parts
        score += len(overlap) * 3.0
        if allowed and lowered.startswith(allowed):
            score += 2.5
        if Path(rel).name.lower() in tokens:
            score += 2.0
        if any(h in lowered for h in _TEST_HINTS):
            score += 0.5
        # Small files are usually the important ones (routers, schemas, config).
        try:
            size = os.path.getsize(os.path.join(self.snapshot.root, rel))
        except OSError:
            size = MAX_FILE_BYTES
        if size < 2000:
            score += 0.8
        elif size > 200_000:
            score -= 2.0
        return score

    def _read(self, rel_paths: list[str]) -> dict[str, str]:
        contents: dict[str, str] = {}
        used = 0
        for rel in rel_paths:
            if used >= self.budget.max_chars:
                break
            full = os.path.join(self.snapshot.root, rel)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                text = Path(full).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) > self.budget.max_file_chars:
                text = text[: self.budget.max_file_chars] + "\n... [truncated]\n"
            contents[rel] = text
            used += len(text)
        return contents

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "add",
            "create", "build", "make", "using", "use", "new", "task", "please",
        }
        words = re.split(r"[^a-z0-9_]+", text.lower())
        return {w for w in words if len(w) > 2 and w not in stop}

    @staticmethod
    def _title_of(task_id: str, results: dict[str, AgentResult]) -> str:
        result = results.get(task_id)
        return result.summary[:120] if result else task_id[:8]

    @staticmethod
    def _summary_of(task_id: str, results: dict[str, AgentResult]) -> str:
        result = results.get(task_id)
        return result.summary if result else ""

    @staticmethod
    def _collect_interfaces(results: dict[str, AgentResult]) -> list[str]:
        interfaces: list[str] = []
        for result in results.values():
            if result.status != TaskStatus.SUCCESS:
                continue
            interfaces.extend(result.output.get("interfaces", []))
        return interfaces[:40]

    @staticmethod
    def _test_requirements(task: str | Task) -> str:
        if isinstance(task, Task) and task.tests:
            return "Required test files: " + ", ".join(task.tests)
        return (
            "Add or update tests for new behaviour. Tests must be deterministic "
            "and must not depend on network access."
        )


def _role_prefixes(task: Task) -> tuple[str, ...]:
    from app.agents.roster import get_role

    if task.assigned_agent:
        try:
            return get_role(task.assigned_agent).allowed_prefixes
        except KeyError:
            return ()
    return ()

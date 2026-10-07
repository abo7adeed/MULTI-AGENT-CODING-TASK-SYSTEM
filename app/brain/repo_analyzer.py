"""
RepositoryAnalyzer.

Answers "what is this codebase?" with facts, not guesses. The deterministic
part walks the tree and inspects manifests; the LLM part, when a provider is
configured, only *interprets* those facts.

That split matters: the LLM never invents structure, it summarises structure
that was measured. If the model is unavailable, the analysis is still useful.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.brain.context import RepositorySnapshot, scan_repository
from app.llm.base import LLMError, LLMProvider
from app.logging_config import get_logger
from app.models.domain import TaskType

logger = get_logger("app.brain.repo_analyzer")

_FRAMEWORK_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fastapi", re.compile(r"\bfrom\s+fastapi\b|\bimport\s+fastapi\b")),
    ("django", re.compile(r"\bfrom\s+django\b|\bimport\s+django\b")),
    ("flask", re.compile(r"\bfrom\s+flask\b|\bimport\s+flask\b")),
    ("sqlalchemy", re.compile(r"\bsqlalchemy\b", re.I)),
    ("alembic", re.compile(r"\balembic\b", re.I)),
    ("pydantic", re.compile(r"\bpydantic\b", re.I)),
    ("langgraph", re.compile(r"\blanggraph\b", re.I)),
    ("langchain", re.compile(r"\blangchain\b", re.I)),
    ("pytest", re.compile(r"\bimport\s+pytest\b|\bpytest\b")),
    ("numpy", re.compile(r"\bnumpy\b", re.I)),
    ("pandas", re.compile(r"\bpandas\b", re.I)),
    ("torch", re.compile(r"\btorch\b", re.I)),
    ("chromadb", re.compile(r"\bchromadb\b", re.I)),
    ("faiss", re.compile(r"\bfaiss\b", re.I)),
)

_JS_FRAMEWORKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("react", re.compile(r"[\"']react[\"']\s*:|from\s+[\"']react[\"']")),
    ("next", re.compile(r"[\"']next[\"']\s*:")),
    ("vue", re.compile(r"[\"']vue[\"']\s*:")),
    ("svelte", re.compile(r"[\"']svelte[\"']\s*:")),
    ("vite", re.compile(r"[\"']vite[\"']\s*:")),
    ("tailwind", re.compile(r"[\"']tailwindcss[\"']|tailwind")),
    ("express", re.compile(r"[\"']express[\"']\s*:")),
    ("vitest", re.compile(r"[\"']vitest[\"']\s*:")),
    ("jest", re.compile(r"[\"']jest[\"']\s*:")),
)


@dataclass
class RepoAnalysis:
    root: str = ""
    is_git_repo: bool = False
    total_files: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    build_tools: list[str] = field(default_factory=list)
    test_frameworks: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dockerised: bool = False
    has_ci: bool = False
    conventions: list[str] = field(default_factory=list)
    directory_map: dict[str, int] = field(default_factory=dict)
    largest_files: list[str] = field(default_factory=list)
    summary: str = ""
    llm_insight: str = ""
    snapshot: Optional[RepositorySnapshot] = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "root": self.root,
            "is_git_repo": self.is_git_repo,
            "total_files": self.total_files,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "build_tools": self.build_tools,
            "test_frameworks": self.test_frameworks,
            "package_managers": self.package_managers,
            "entry_points": self.entry_points,
            "test_files": self.test_files[:30],
            "config_files": self.config_files,
            "dockerised": self.dockerised,
            "has_ci": self.has_ci,
            "conventions": self.conventions,
            "directory_map": self.directory_map,
            "summary": self.summary,
            "llm_insight": self.llm_insight,
        }
        return data

    @property
    def primary_language(self) -> str:
        if not self.languages:
            return "unknown"
        return max(self.languages.items(), key=lambda kv: kv[1])[0]

    def capability_hints(self) -> list[TaskType]:
        """Which task types this repository can plausibly support."""
        hints: list[TaskType] = []
        fw = set(self.frameworks)
        if fw & {"fastapi", "django", "flask", "express"}:
            hints.append(TaskType.BACKEND)
        if fw & {"react", "next", "vue", "svelte"}:
            hints.append(TaskType.FRONTEND)
        if fw & {"sqlalchemy", "alembic", "prisma"} or "database" in fw:
            hints.append(TaskType.DATABASE)
        if fw & {"langgraph", "langchain", "torch", "chromadb", "faiss"}:
            hints.append(TaskType.AI_ML)
        if self.test_files or fw & {"pytest", "jest", "vitest"}:
            hints.append(TaskType.TESTING)
        if self.dockerised or self.has_ci:
            hints.append(TaskType.DEVOPS)
        if self.config_files:
            hints.append(TaskType.DOCUMENTATION)
        return hints


class RepositoryAnalyzer:
    """
    Produces a `RepoAnalysis` for a project directory.

        analysis = await RepositoryAnalyzer().analyze("/path/to/repo")
    """

    def __init__(self, llm: Optional[LLMProvider] = None, max_files: int = 3000):
        self.llm = llm
        self.max_files = max_files

    async def analyze(self, repo_path: str | Path, use_llm: bool = True) -> RepoAnalysis:
        root = Path(repo_path).expanduser()
        snapshot = scan_repository(root, max_files=self.max_files)
        analysis = self._analyse_deterministically(root, snapshot)

        if use_llm and self.llm is not None:
            analysis.llm_insight = await self._enrich(analysis)
        return analysis

    # ── deterministic core ──────────────────────────────────────────────────

    def _analyse_deterministically(
        self, root: Path, snapshot: RepositorySnapshot
    ) -> RepoAnalysis:
        analysis = RepoAnalysis(root=str(root), snapshot=snapshot)
        analysis.total_files = snapshot.total_files
        analysis.languages = dict(snapshot.languages)
        analysis.entry_points = snapshot.entry_points
        analysis.test_files = snapshot.test_files
        analysis.config_files = snapshot.config_files
        analysis.conventions = list(snapshot.conventions)
        analysis.is_git_repo = (root / ".git").exists()

        config_names = {Path(c).name.lower() for c in snapshot.config_files}

        analysis.dockerised = any(n.startswith("dockerfile") for n in config_names) or (
            "docker-compose.yml" in config_names or "docker-compose.yaml" in config_names
        )
        analysis.has_ci = self._detect_ci(root)

        if config_names & {"pyproject.toml", "requirements.txt", "setup.py", "pipfile"}:
            analysis.package_managers.append("pip")
        if config_names & {"package.json", "pnpm-lock.yaml", "yarn.lock", "package-lock.json"}:
            analysis.package_managers.append("npm")
        if "go.mod" in config_names:
            analysis.package_managers.append("go")
        if "cargo.toml" in config_names:
            analysis.package_managers.append("cargo")

        # Framework detection: manifests first (cheap, reliable), then content.
        manifests = self._read_manifests(root, config_names)
        blob = manifests + self._sample_sources(root, snapshot)
        for name, pattern in _FRAMEWORK_MARKERS:
            if pattern.search(blob):
                analysis.frameworks.append(name)
        for name, pattern in _JS_FRAMEWORKS:
            if pattern.search(blob):
                analysis.frameworks.append(name)

        analysis.frameworks = sorted(set(analysis.frameworks))
        analysis.test_frameworks = [
            f for f in analysis.frameworks if f in {"pytest", "jest", "vitest", "unittest"}
        ]
        analysis.build_tools = [
            f
            for f in analysis.frameworks
            if f in {"vite", "next", "webpack", "setuptools", "poetry", "alembic"}
        ] or (
            ["pip/pyproject"] if "pyproject.toml" in config_names else []
        )

        analysis.directory_map = self._directory_map(snapshot)
        analysis.largest_files = self._largest_files(root, snapshot)
        analysis.summary = self._summarise(analysis)
        return analysis

    @staticmethod
    def _detect_ci(root: Path) -> bool:
        """
        Look for CI definitions on disk.

        The repository walk deliberately skips dot-directories, so
        `.github/workflows/` never appears in the snapshot and cannot be used
        here: relying on it made `has_ci` permanently False for every real repo.
        """
        for workflows in (
            root / ".github" / "workflows",
            root / ".gitlab",
            root / ".circleci",
            root / ".azure-pipelines",
        ):
            if workflows.is_dir() and any(workflows.iterdir()):
                return True
        for name in (
            "azure-pipelines.yml",
            "azure-pipelines.yaml",
            ".travis.yml",
            "Jenkinsfile",
            ".drone.yml",
        ):
            if (root / name).exists():
                return True
        return False

    def _read_manifests(self, root: Path, config_names: set[str]) -> str:
        chunks: list[str] = []
        for name in sorted(config_names):
            if not name.endswith((".toml", ".json", ".cfg", ".txt", ".ini", ".yaml", ".yml")):
                continue
            path = root / name
            if not path.is_file():
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace")[:8000])
            except OSError:
                continue
        return "\n".join(chunks)

    def _sample_sources(self, root: Path, snapshot: RepositorySnapshot) -> str:
        """A bounded sample of source files, for framework keyword detection."""
        sample = [
            f
            for f in snapshot.files
            if Path(f).suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".json"}
        ][:40]
        chunks: list[str] = []
        for rel in sample:
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="replace")[:2000]
            except OSError:
                continue
            chunks.append(text)
            if sum(len(c) for c in chunks) > 60_000:
                break
        return "\n".join(chunks)

    @staticmethod
    def _directory_map(snapshot: RepositorySnapshot) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rel in snapshot.files:
            top = rel.split("/", 1)[0] if "/" in rel else "."
            counts[top] = counts.get(top, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:15])

    @staticmethod
    def _largest_files(root: Path, snapshot: RepositorySnapshot) -> list[str]:
        sizes: list[tuple[int, str]] = []
        for rel in snapshot.files[:600]:
            try:
                sizes.append((( root / rel).stat().st_size, rel))
            except OSError:
                continue
        sizes.sort(reverse=True)
        return [rel for _, rel in sizes[:10]]

    @staticmethod
    def _summarise(analysis: RepoAnalysis) -> str:
        if analysis.total_files == 0:
            return "The repository is empty or does not exist yet; this is a greenfield build."
        primary = analysis.primary_language
        parts = [
            f"{analysis.total_files} files, primarily {primary}",
            f"frameworks: {', '.join(analysis.frameworks) or 'none detected'}",
        ]
        if analysis.entry_points:
            parts.append(f"entry points: {', '.join(analysis.entry_points[:3])}")
        if analysis.test_files:
            parts.append(f"{len(analysis.test_files)} test file(s)")
        if analysis.dockerised:
            parts.append("containerised")
        if analysis.has_ci:
            parts.append("has CI")
        return "; ".join(parts)

    # ── optional LLM interpretation ─────────────────────────────────────────

    async def _enrich(self, analysis: RepoAnalysis) -> str:
        """Ask the model to interpret the measured facts. Failure is non-fatal."""
        from app.llm.opencode import parse_json_response

        facts = json.dumps(
            {
                "languages": analysis.languages,
                "frameworks": analysis.frameworks,
                "entry_points": analysis.entry_points[:10],
                "top_directories": analysis.directory_map,
                "conventions": analysis.conventions,
                "test_files": analysis.test_files[:10],
            },
            indent=2,
        )
        system = (
            "You are a senior engineer surveying an unfamiliar repository. "
            "You are given measured facts. Interpret them for someone who must "
            "extend this codebase today. Be concrete and brief."
        )
        user = (
            f"Repository facts:\n{facts}\n\n"
            "Answer with JSON: "
            '{"architecture": "...", "extension_points": ["..."], '
            '"risks": ["..."], "relevant_modules": ["path", "..."]}'
        )
        try:
            response = await self.llm.complete(user, system)
        except (LLMError, Exception) as exc:  # noqa: BLE001
            logger.info("Repository enrichment skipped", extra={"reason": str(exc)[:200]})
            return ""

        parsed = parse_json_response(response.text)
        if isinstance(parsed, dict):
            return json.dumps(
                {
                    "architecture": parsed.get("architecture", ""),
                    "extension_points": parsed.get("extension_points", []),
                    "risks": parsed.get("risks", []),
                    "relevant_modules": parsed.get("relevant_modules", []),
                },
                indent=2,
            )
        return response.text.strip()[:2000]

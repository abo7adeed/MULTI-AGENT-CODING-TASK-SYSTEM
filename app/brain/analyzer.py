"""
TaskAnalyzer.

Turns a free-form request into a structured specification. Deterministic signal
extraction runs first (keywords, implied deliverables, ambiguity flags); the LLM
is asked to enrich that specification, never to invent it from nothing.

The point is that the same analyzer works with or without a model, so a run
never fails merely because the provider is down.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.llm.base import LLMError, LLMProvider
from app.logging_config import get_logger
from app.models.domain import TaskType

logger = get_logger("app.brain.analyzer")

#: Phrase -> task type. Ordered: first match wins for each distinct type.
_INTENT_RULES: tuple[tuple[re.Pattern[str], TaskType], ...] = (
    (re.compile(r"\b(rag|retrieval[- ]augmented|vector (store|search|db)|embedding\w*)\b", re.I), TaskType.AI_ML),
    (re.compile(r"\b(llm|openai|anthropic|model|prompt|agents?)\b", re.I), TaskType.AI_ML),
    (re.compile(r"\b(react|vue|angular|svelte|frontend|front-end|ui|dashboard)\b", re.I), TaskType.FRONTEND),
    (re.compile(r"\b(auth\w*|login|jwt|oauth|sso|sessions?|password|rbac)\b", re.I), TaskType.SECURITY),
    (re.compile(r"\b(api|endpoint|rest|graphql|backend|back-end|server|microservice)\b", re.I), TaskType.BACKEND),
    (re.compile(r"\b(databases?|postgres\w*|sql|schema|migrat\w*|orm|redis|mongo)\b", re.I), TaskType.DATABASE),
    (re.compile(r"\b(docker\w*|containers?|kubernetes|helm|terraform|ci/cd|pipelines?|deploy\w*)\b", re.I), TaskType.DEVOPS),
    (re.compile(r"\b(test\w*|coverage|pytest|jest|spec\w*|qa)\b", re.I), TaskType.TESTING),
    (re.compile(r"\b(document\w*|readme|docs|guide|tutorial|changelog)\b", re.I), TaskType.DOCUMENTATION),
    (re.compile(r"\b(refactor\w*|restructure\w*|clean ?up|tech debt|modularis\w*|modulariz\w*)\b", re.I), TaskType.REFACTOR),
    (re.compile(r"\b(architecture|architect|design|structure|modular)\b", re.I), TaskType.ARCHITECTURE),
    (re.compile(r"\b(plan\w*|roadmap|break ?down|decompose|design doc)\b", re.I), TaskType.PLANNING),
)

_DELIVERABLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(api|endpoint|rest|backend|service)\b", re.I), "backend API"),
    (re.compile(r"\b(frontend|ui|react|page|dashboard|interface)\b", re.I), "user interface"),
    (re.compile(r"\b(test|testing|pytest|jest|coverage|spec)\w*", re.I), "automated tests"),
    (re.compile(r"\bdocker\w*|\bcontaineris\w*", re.I), "container setup"),
    (re.compile(r"\b(ci|cd|pipeline|workflow)\b|github actions", re.I), "CI/CD pipeline"),
    (re.compile(r"\b(auth|authentication|login|jwt|oauth)\w*", re.I), "authentication"),
    (re.compile(r"\b(rag|retrieval\w*|vector|embedding\w*)\b", re.I), "retrieval pipeline"),
    (re.compile(r"\b(document\w*|readme|docs)\b", re.I), "documentation"),
    (re.compile(r"\b(deploy\w*|hosting|cloud|aws|gcp|azure)\b", re.I), "deployment config"),
    (re.compile(r"\b(migrat\w*|schema|databases?|postgres\w*|sql|orm|tables?)\b", re.I), "data layer"),
)

_AMBIGUITY_MARKERS = (
    "maybe", "probably", "something like", "or something", "etc", "and so on",
    "you decide", "your call", "whatever", "some kind of",
)


@dataclass
class TaskSpecification:
    """Structured understanding of what the user asked for."""

    raw_request: str = ""
    title: str = ""
    intent: str = ""
    task_types: list[TaskType] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    is_greenfield: bool = False
    complexity: str = "moderate"
    llm_plan: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_request": self.raw_request,
            "title": self.title,
            "intent": self.intent,
            "task_types": [t.value for t in self.task_types],
            "deliverables": self.deliverables,
            "keywords": self.keywords,
            "constraints": self.constraints,
            "ambiguities": self.ambiguities,
            "acceptance_criteria": self.acceptance_criteria,
            "is_greenfield": self.is_greenfield,
            "complexity": self.complexity,
            "llm_plan": self.llm_plan,
            "source": self.source,
        }

    @property
    def primary_type(self) -> TaskType:
        return self.task_types[0] if self.task_types else TaskType.GENERIC


_STOPWORDS = {
    "build", "create", "make", "add", "implement", "write", "develop", "generate",
    "please", "with", "and", "the", "a", "an", "for", "that", "this", "using",
    "into", "from", "application", "app", "project", "system", "code", "new",
    "then", "also", "should", "must", "can", "will", "have", "has", "need",
}


class TaskAnalyzer:
    """
    Understands the request. Heuristic first, LLM enrichment optional.

        spec = await TaskAnalyzer().analyze("Build a RAG app with auth")
    """

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm

    async def analyze(
        self,
        request: str,
        repo_analysis: Optional[dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> TaskSpecification:
        spec = self._heuristic(request, repo_analysis)
        if use_llm and self.llm is not None:
            spec.llm_plan = await self._enrich(spec, repo_analysis)
            if spec.llm_plan:
                spec.source = "llm+heuristic"
        return spec

    # ── deterministic core ──────────────────────────────────────────────────

    def _heuristic(
        self, request: str, repo_analysis: Optional[dict[str, Any]] = None
    ) -> TaskSpecification:
        text = request.strip()
        spec = TaskSpecification(raw_request=text)

        spec.task_types = self._detect_types(text)
        spec.deliverables = self._detect_deliverables(text)
        spec.keywords = self._keywords(text)
        spec.ambiguities = self._detect_ambiguities(text)
        spec.title = self._title(text)

        # Must be settled before `_intent`, which phrases a greenfield request
        # differently from a change to an existing codebase.
        repo_analysis = repo_analysis or {}
        total = int(repo_analysis.get("total_files", 0) or 0)
        spec.is_greenfield = total == 0
        spec.intent = self._intent(text, spec)

        spec.acceptance_criteria = self._acceptance_criteria(spec, repo_analysis)
        spec.constraints = self._constraints(text, repo_analysis)
        spec.complexity = self._complexity(spec)
        return spec

    @staticmethod
    def _detect_types(text: str) -> list[TaskType]:
        found: list[TaskType] = []
        for pattern, task_type in _INTENT_RULES:
            if pattern.search(text) and task_type not in found:
                found.append(task_type)
        if TaskType.ANALYSIS not in found and found:
            found.insert(0, TaskType.ANALYSIS)
        if not found:
            found = [TaskType.ANALYSIS, TaskType.PLANNING, TaskType.GENERIC]
        return found

    @staticmethod
    def _detect_deliverables(text: str) -> list[str]:
        seen: list[str] = []
        for pattern, label in _DELIVERABLE_RULES:
            if pattern.search(text) and label not in seen:
                seen.append(label)
        return seen

    @staticmethod
    def _keywords(text: str) -> list[str]:
        words = re.split(r"[^A-Za-z0-9_+#.-]+", text)
        counts: dict[str, int] = {}
        for word in words:
            lowered = word.lower().strip(".-")
            if len(lowered) < 3 or lowered in _STOPWORDS or lowered.isdigit():
                continue
            counts[lowered] = counts.get(lowered, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ranked[:25]]

    @staticmethod
    def _detect_ambiguities(text: str) -> list[str]:
        found: list[str] = []
        lowered = text.lower()
        for marker in _AMBIGUITY_MARKERS:
            if marker in lowered:
                found.append(f"request contains the vague phrase '{marker.strip()}'")
        if not re.search(r"\b(api|endpoint|interface|schema|function|class)\b", text, re.I):
            found.append("no explicit interface or API was specified")
        return found

    @staticmethod
    def _title(text: str) -> str:
        cleaned = re.sub(r"^\s*(please\s+)?", "", text, flags=re.I).strip()
        first_clause = re.split(r"[.\n;]", cleaned)[0].strip()
        if len(first_clause) > 90:
            first_clause = first_clause[:87].rstrip() + "..."
        return first_clause or text[:80]

    def _intent(self, text: str, spec: TaskSpecification) -> str:
        if spec.is_greenfield:
            return f"Greenfield build: {', '.join(d for d in spec.deliverables) or 'an application'}."
        verbs = [v for v in ("add", "extend", "fix", "refactor", "migrate", "optimise", "optimize")
                 if re.search(rf"\b{v}\b", text, re.I)]
        return (
            f"Modify the existing codebase to {', '.join(verbs) or 'implement'} "
            f"{', '.join(spec.deliverables) or 'the requested change'}."
        )

    def _acceptance_criteria(self, spec: TaskSpecification, repo: dict[str, Any]) -> list[str]:
        criteria: list[str] = []
        for deliverable in spec.deliverables:
            criteria.append(f"{deliverable} is implemented and importable")
        if "automated tests" in spec.deliverables:
            criteria.append("Automated tests exist for the new behaviour and pass")
        else:
            criteria.append("Existing test suite still passes")
        if repo.get("test_frameworks"):
            criteria.append(f"Tests run under the project's existing {repo['test_frameworks'][0]} setup")
        if "container setup" in spec.deliverables:
            criteria.append("The container image builds and starts")
        criteria.append("No secrets or credentials are committed")
        return criteria

    def _constraints(self, text: str, repo: dict[str, Any]) -> list[str]:
        constraints: list[str] = []
        explicit = re.findall(r"\b(must|should|without|no|never|only|using)\s+([^.,;]+)", text, re.I)
        for _, clause in explicit[:6]:
            clause = clause.strip()
            if len(clause) > 3:
                constraints.append(clause[0].upper() + clause[1:])
        if repo.get("conventions"):
            constraints.append("Follow existing repository conventions")
        if repo.get("package_managers"):
            constraints.append(f"Use the existing package manager: {', '.join(repo['package_managers'])}")
        if repo.get("python_version"):
            constraints.append(f"Target Python {repo['python_version']}")
        return constraints

    @staticmethod
    def _complexity(spec: TaskSpecification) -> str:
        score = len(spec.deliverables) * 2 + len(spec.task_types)
        if score <= 5:
            return "simple"
        if score <= 10:
            return "moderate"
        if score <= 16:
            return "complex"
        return "very large"

    # ── optional LLM enrichment ─────────────────────────────────────────────

    async def _enrich(
        self, spec: TaskSpecification, repo: Optional[dict[str, Any]]
    ) -> str:
        from app.llm.opencode import parse_json_response

        system = (
            "You are a senior technical lead. Given a feature request and a summary "
            "of an existing codebase, produce a concrete implementation plan. "
            "Be specific: name real files and modules. Reply with JSON only."
        )
        user = json.dumps(
            {
                "request": spec.raw_request,
                "detected_task_types": [t.value for t in spec.task_types],
                "detected_deliverables": spec.deliverables,
                "detected_ambiguities": spec.ambiguities,
                "repository": {
                    k: repo.get(k)
                    for k in ("total_files", "languages", "frameworks", "entry_points", "test_files")
                }
                if repo
                else {},
            },
            indent=2,
        )
        try:
            response = await self.llm.complete(
                user + "\n\nReply with JSON: "
                '{"summary": "...", "steps": ["..."], "risks": ["..."], '
                '"acceptance_criteria": ["..."]}',
                system,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Task analysis enrichment skipped", extra={"reason": str(exc)[:200]})
            return ""

        parsed = parse_json_response(response.text)
        if isinstance(parsed, dict):
            spec.acceptance_criteria.extend(
                c for c in parsed.get("acceptance_criteria", []) if c not in spec.acceptance_criteria
            )
            return json.dumps(
                {
                    "summary": parsed.get("summary", ""),
                    "steps": parsed.get("steps", []),
                    "risks": parsed.get("risks", []),
                },
                indent=2,
            )
        return response.text.strip()[:3000]

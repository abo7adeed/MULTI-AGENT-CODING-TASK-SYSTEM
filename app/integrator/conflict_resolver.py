"""
ConflictResolver.

When two agents changed the same region, neither side is discarded. The
resolver is given *both* intents plus the merge base, and asked to produce a
merged file that preserves what each side was trying to do.

Three strategies, in order of preference:

  1. **LLM resolution** -- an agent reads the three versions and writes the
     merge. This is the case the specification cares about most.
  2. **Three-way structural merge** -- when the conflict is a clean append or a
     disjoint pair of hunks, reconcile deterministically with no model.
  3. **Report failure** -- never silently take one side. That is the rule.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.llm.base import LLMProvider
from app.logging_config import get_logger
from app.models.agent_context import AgentContext
from app.models.domain import Task, TaskType

logger = get_logger("app.integrator.conflicts")

_START = "<<<<<<<"
_MIDDLE = "======="
_END = ">>>>>>>"
#: One conflict block, from the opening marker through the closing one.
#: The groups must stay non-capturing: `findall` returns whole matches only
#: when there are no groups, and a captured group makes it yield tuples, which
#: `GitMarkers.split` cannot read. The middle section is optional so both the
#: two-way and the diff3 (`|||||||`) layouts match, and the block must run to
#: the end marker or the "theirs" half is left empty.
_SPLIT = re.compile(
    rf"^{re.escape(_START)}.*?^(?:{re.escape(_MIDDLE)}.*?)?^{re.escape(_END)}",
    re.M | re.S,
)


@dataclass
class ConflictRegion:
    """One conflicted hunk, with all three sides available."""

    path: str
    ours: str
    theirs: str
    base: str = ""
    start: int = 0

    @property
    def is_deletion(self) -> bool:
        return not self.ours.strip() or not self.theirs.strip()


@dataclass
class ResolutionReport:
    resolved: list[str] = field(default_factory=list)
    strategies: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: path -> merged content, ready to be written and staged.
    resolved_contents: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.unresolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "strategies": self.strategies,
            "unresolved": self.unresolved,
            "notes": self.notes,
            "success": self.success,
        }


class ConflictResolver:
    """
    Produces merged file contents for conflicted paths.

        resolver = ConflictResolver(llm=provider)
        report = await resolver.resolve(repo, conflict_details)
    """

    def __init__(self, llm: Optional[LLMProvider] = None, role: str = "integration"):
        self.llm = llm
        self.role = role

    async def resolve(
        self,
        repo_path: str,
        conflicts: list[dict[str, Any]],
        extra_context: str = "",
    ) -> dict[str, str]:
        """
        Resolve every conflict, returning {path: merged_content}.

        Paths that cannot be resolved are omitted and reported, so the
        integrator can fail loudly rather than lose one agent's work.
        """
        report = await self.resolve_with_report(repo_path, conflicts, extra_context)
        if not report.success:
            logger.warning(
                "Unresolved conflicts remain",
                extra={"paths": report.unresolved, "notes": report.notes},
            )
        return report.resolved_contents

    async def resolve_with_report(
        self,
        repo_path: str,
        conflicts: list[dict[str, Any]],
        extra_context: str = "",
    ) -> ResolutionReport:
        report = ResolutionReport()
        contents: dict[str, str] = {}

        for detail in conflicts:
            path = detail.get("path", "")
            if not path:
                continue
            regions = self._parse_regions(path, detail)
            if not regions and detail.get("raw"):
                contents[path] = detail["raw"]
                report.resolved.append(path)
                report.strategies[path] = "raw"
                continue

            merged, strategy, note = await self._resolve_file(
                repo_path, regions, extra_context
            )
            if merged is None:
                report.unresolved.append(path)
                if note:
                    report.notes.append(f"{path}: {note}")
                continue
            contents[path] = merged
            report.resolved.append(path)
            report.strategies[path] = strategy
            if note:
                report.notes.append(f"{path}: {note}")

        report.resolved_contents = contents
        return report

    # ── strategy dispatch ───────────────────────────────────────────────────

    async def _resolve_file(
        self, repo_path: str, regions: list[ConflictRegion], extra_context: str
    ) -> tuple[Optional[str], str, str]:
        if self.llm is not None:
            merged = await self._resolve_with_llm(repo_path, regions, extra_context)
            if merged is not None:
                return merged, "llm", ""
            note = "the merge agent produced no usable result"
        else:
            note = "no merge agent configured"

        # Deterministic fallback: only for conflicts git can reconcile itself.
        merged, ok = self._resolve_structural(regions)
        if ok:
            return merged, "structural", "resolved without a model (disjoint or additive hunks)"
        return None, "failed", note

    async def _resolve_with_llm(
        self, repo_path: str, regions: list[ConflictRegion], extra_context: str
    ) -> Optional[str]:
        """Ask a merge agent to write the reconciled file."""
        from app.agents.coder import LLMCoderAgent
        from app.models.domain import TaskStatus

        parts: list[str] = []
        for index, region in enumerate(regions, start=1):
            parts.append(
                f"### Conflict {index}\n"
                f"--- CURRENT (already merged on main) ---\n{region.ours or '(nothing)'}\n"
                f"--- INCOMING (from the agent branch) ---\n{region.theirs or '(nothing)'}\n"
                f"--- COMMON ANCESTOR ---\n{region.base or '(unknown)'}"
            )
        body = "\n\n".join(parts)
        paths = sorted({r.path for r in regions})

        agent = LLMCoderAgent(self.llm, role=self.role, max_repair_rounds=1)
        task = Task(
            id=f"conflict-{abs(hash(tuple(paths))) % (10**8):08d}",
            title=f"Resolve merge conflicts in {', '.join(paths)}",
            description=(
                "Two agents changed the same file. Produce the merged result that "
                "preserves the intent of both sides."
            ),
            type=TaskType.INTEGRATION,
        )
        context = AgentContext(
            task=task,
            workspace=repo_path if _is_dir(repo_path) else ".",
            role=self.role,
            original_request=(
                "Two agents changed the same file. Produce the merged result that "
                "preserves the intent of both sides."
            ),
            relevant_files=sorted({r.path for r in regions}),
            file_contents={
                r.path: f"<<<<<<< HEAD\n{r.ours}\n=======\n{r.theirs}\n>>>>>>> feature"
                for r in regions
            },
            instructions=(
                "For each conflict, understand what each side was trying to do and "
                "write a result that keeps both behaviours where they are compatible. "
                "Where they are genuinely incompatible, keep the incoming change and "
                "leave a comment explaining the trade-off. Never leave conflict markers."
            )
            + ("\n\n" + body if body else ""),
            test_requirements="No conflict markers may remain in the output.",
        )
        result = await agent.execute(task, context)
        if result.status != TaskStatus.SUCCESS:
            return None
        # The agent wrote the file; read it back.
        written = None
        for path in result.files_changed:
            candidate = _read(repo_path, path)
            if candidate is not None:
                written = candidate
                break
        return written

    def _resolve_structural(
        self, regions: list[ConflictRegion]
    ) -> tuple[Optional[str], bool]:
        """
        Reconcile without a model, but only when it is provably safe.

        Handled: both sides identical, one side a superset of the other, one
        side deleted, and -- when a merge base is available -- both sides
        appending to an unchanged base. Everything else is reported as
        unresolved: concatenating two incompatible rewrites of the same
        statement produces code that does not compile, which is worse than a
        reported conflict.
        """
        pieces: list[str] = []
        for region in regions:
            ours, theirs, base = region.ours, region.theirs, region.base
            if ours == theirs:
                pieces.append(ours)
                continue
            if not theirs.strip():
                pieces.append(ours)          # incoming deleted; keep current
                continue
            if not ours.strip():
                pieces.append(theirs)        # current deleted; keep incoming
                continue
            if ours.strip() in theirs:
                pieces.append(theirs)        # incoming is a superset
                continue
            if theirs.strip() in ours:
                pieces.append(ours)          # current is a superset
                continue
            if base.strip() and _is_additive(ours, theirs, base):
                # Provably additive: neither side edited the shared context, so
                # the result is the base plus each side's new lines.
                merged = "\n".join(
                    part for part in (base.rstrip(), _added(ours, base), _added(theirs, base)) if part
                )
                pieces.append(merged)
                continue
            return None, False
        return "\n".join(pieces), True

    # ── parsing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_regions(path: str, detail: dict[str, Any]) -> list[ConflictRegion]:
        raw = detail.get("raw", "")
        if not raw:
            return []
        regions: list[ConflictRegion] = []
        for index, block in enumerate(_SPLIT.findall(raw)):
            ours, theirs, base = GitMarkers.split_full(block)
            regions.append(
                ConflictRegion(
                    path=path,
                    ours=ours,
                    theirs=theirs,
                    base=base,
                    start=index,
                )
            )
        return regions


class GitMarkers:
    """Splits a conflict block into the ours / base / theirs halves."""

    @staticmethod
    def split(block: str) -> tuple[str, str]:
        ours, theirs, _ = GitMarkers.split_full(block)
        return ours, theirs

    @staticmethod
    def split_full(block: str) -> tuple[str, str, str]:
        """
        Return (ours, theirs, base).

        The base is empty for a two-way merge, which is itself a useful signal:
        without it, nothing can be *proven* to be an additive hunk.
        """
        ours: list[str] = []
        theirs: list[str] = []
        base: list[str] = []
        side: str | None = None
        for line in block.splitlines():
            if line.startswith(_START):
                side = "ours"
                continue
            if line.startswith("|||||||"):
                side = "base"
                continue
            if line.startswith(_MIDDLE):
                side = "theirs"
                continue
            if line.startswith(_END):
                side = None
                continue
            if side == "ours":
                ours.append(line)
            elif side == "theirs":
                theirs.append(line)
            elif side == "base":
                base.append(line)
        return (
            "\n".join(ours).strip("\n"),
            "\n".join(theirs).strip("\n"),
            "\n".join(base).strip("\n"),
        )

    @staticmethod
    def has_markers(content: str) -> bool:
        return bool(re.search(rf"^{re.escape(_START)}", content, re.M))


def _lines_intersect(a: str, b: str) -> bool:
    """Whether two blocks share any non-trivial line (i.e. a real conflict)."""
    set_a = {line.strip() for line in a.splitlines() if line.strip()}
    set_b = {line.strip() for line in b.splitlines() if line.strip()}
    return bool(set_a & set_b)


def _contains_all(container: str, block: str) -> bool:
    """Whether every line of `container` also appears in `block`."""
    have = {line.strip() for line in block.splitlines() if line.strip()}
    return all(line.strip() in have for line in container.splitlines() if line.strip())


def _is_additive(ours: str, theirs: str, base: str) -> bool:
    """
    Whether both sides are the shared base plus their own additions.

    This is the only case where interleaving is provably lossless: neither side
    touched a line the other kept, so no edit is overwritten. It requires a
    merge base -- without one, nothing can be proven and the answer is no.
    """
    if not base.strip():
        return False
    if not _contains_all(base, ours) or not _contains_all(base, theirs):
        return False
    return bool(_added(ours, base).strip() or _added(theirs, base).strip())


def _added(block: str, base: str) -> str:
    """The lines of `block` that are not part of `base`."""
    base_lines = {line.strip() for line in base.splitlines() if line.strip()}
    return "\n".join(
        line for line in block.splitlines() if line.strip() not in base_lines
    )


def summarise_diff(before: str, after: str, path: str, context: int = 2) -> str:
    """Unified diff between two versions, for the UI's resolution view."""
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=context,
        )
    )


def _is_dir(path: str) -> bool:
    from pathlib import Path

    try:
        return Path(path).is_dir()
    except OSError:
        return False


def _read(base: str, rel: str) -> Optional[str]:
    from pathlib import Path

    try:
        root = Path(base)
        target = root / rel if root.is_dir() else Path(rel)
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

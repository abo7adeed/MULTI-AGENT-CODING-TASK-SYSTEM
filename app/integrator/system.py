"""
Integrator.

Takes the branches the agents produced and turns them into one verified
repository state. The order matters and is not negotiable:

    collect -> snapshot baseline -> merge (resolving conflicts) -> test
            -> if tests fail, one repair round -> review -> report

Two rules define the behaviour:

  * **A conflict never silently picks a winner.** Either a merge agent
    reconciles both intents, or the branch is left unmerged and reported.
  * **A failed integration rolls back** to the snapshot taken before any
    merge, so a partial merge is never left behind for the user to untangle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import Settings, get_settings
from app.events import EventBus, EventType, get_event_bus
from app.git.manager import GitManager, MergeOutcome
from app.integrator.conflict_resolver import ConflictResolver
from app.integrator.test_runner import TestRunner
from app.llm.base import LLMProvider
from app.logging_config import get_logger
from app.models.domain import (
    AgentResult,
    ExecutionState,
    FinalStatus,
    IntegrationReport,
    TaskStatus,
    TaskType,
    TestRunResult,
)

logger = get_logger("app.integrator")

#: Task kinds whose whole job is to read and report, not to change code. Their
#: agents legitimately commit nothing, so an empty branch is not a defect.
_READ_ONLY_TASK_TYPES = frozenset({TaskType.ANALYSIS.value, TaskType.REVIEW.value})


@dataclass
class ChangeRecord:
    """What one agent contributed, as the integrator sees it."""

    task_id: str
    title: str
    branch: Optional[str]
    commit: Optional[str]
    files_changed: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    status: str = "SUCCESS"
    skip_reason: str = ""
    task_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "branch": self.branch,
            "commit": self.commit,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "task_type": self.task_type,
        }


class ChangeCollector:
    """Groups successful agent results into an ordered list of branches."""

    async def collect(
        self, state: ExecutionState, git: GitManager
    ) -> tuple[list[ChangeRecord], list[ChangeRecord]]:
        """
        Return (mergeable, skipped).

        Ordered by the DAG topological order so a branch always merges after
        the branches it was built on top of. Every skipped record carries a
        reason, because "9 tasks succeeded but 0 branches were merged" must
        never be reported as a clean success.
        """
        from app.engine.dag import DAGEngine

        engine = DAGEngine(state.dag)
        order = engine.topological_order() if state.dag.tasks else list(state.agent_results)

        mergeable: list[ChangeRecord] = []
        skipped: list[ChangeRecord] = []
        for task_id in order:
            result: AgentResult | None = state.agent_results.get(task_id)
            if result is None:
                continue
            task = state.tasks.get(task_id)
            branch = result.branch or (task.branch if task else None)
            record = ChangeRecord(
                task_id=task_id,
                title=task.title if task else task_id[:8],
                branch=branch,
                commit=result.commit,
                files_changed=list(result.files_changed),
                tests_passed=result.tests_passed,
                tests_failed=result.tests_failed,
                status=result.status.value,
                task_type=task.type.value if task else "",
            )
            if result.status != TaskStatus.SUCCESS:
                record.skip_reason = f"task ended as {result.status.value}"
            elif not branch:
                record.skip_reason = "agent produced no branch (no git workspace)"
            elif not await git.branch_exists(branch):
                record.skip_reason = f"branch {branch} no longer exists"
            elif not record.commit and not record.files_changed:
                record.skip_reason = "agent made no file changes"
                skipped.append(record)
                continue
            else:
                mergeable.append(record)
                continue
            skipped.append(record)
        return mergeable, skipped


class RegressionDetector:
    """
    Compares a test run against the baseline taken before merging.

    Distinguishes "these tests were already failing" from "this change broke
    them", which is the difference between a real regression and noise.
    """

    def __init__(self, failure_threshold: int = 0):
        self.failure_threshold = failure_threshold

    def compare(
        self, baseline: Optional[TestRunResult], after: TestRunResult
    ) -> list[str]:
        if not baseline or baseline.skipped or after.skipped:
            return []
        findings: list[str] = []
        if baseline.passed and not after.passed:
            findings.append(
                f"Regression: the suite passed before integration and fails after "
                f"(exit {after.exit_code}). {after.summary}"
            )
        delta = after.failed_count - baseline.failed_count
        if delta > self.failure_threshold:
            findings.append(
                f"Regression: failing tests increased by {delta} "
                f"({baseline.failed_count} -> {after.failed_count})"
            )
        if baseline.passed_count and after.passed_count < baseline.passed_count:
            findings.append(
                f"Coverage regression: passing tests dropped from "
                f"{baseline.passed_count} to {after.passed_count}"
            )
        return findings


class FinalReviewer:
    """Optional LLM review of the integrated diff. Read-only, never blocks."""

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm

    async def review(self, diff: str, max_chars: int = 12_000) -> list[dict[str, Any]]:
        if self.llm is None or not diff.strip():
            return []
        from app.llm.opencode import parse_json_response

        system = (
            "You are a meticulous staff engineer reviewing an integrated change. "
            "Report only concrete, actionable findings. Reply with JSON only."
        )
        user = (
            f"Diff under review:\n\n{diff[:max_chars]}\n\n"
            'Reply with JSON: {"findings": [{"severity": "critical|major|minor", '
            '"file": "...", "line": 0, "issue": "...", "fix": "..."}]}'
        )
        try:
            response = await self.llm.complete(user, system)
        except Exception as exc:  # noqa: BLE001 - review is advisory
            logger.info("Final review skipped", extra={"reason": str(exc)[:200]})
            return []
        parsed = parse_json_response(response.text)
        if isinstance(parsed, dict) and isinstance(parsed.get("findings"), list):
            return [f for f in parsed["findings"] if isinstance(f, dict)]
        return [{"severity": "info", "issue": response.text.strip()[:1000], "file": "", "line": 0}]


class Integrator:
    """
    Merges, resolves, tests and reviews the agents' work.

        integrator = Integrator(git, settings=settings)
        report = await integrator.integrate(state)
    """

    def __init__(
        self,
        git_manager: GitManager,
        test_runner: Optional[TestRunner] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        reviewer: Optional[FinalReviewer] = None,
        settings: Optional[Settings] = None,
        event_bus: Optional[EventBus] = None,
        orchestration_id: str = "",
        llm: Optional[LLMProvider] = None,
    ):
        self.settings = settings or get_settings()
        self.git = git_manager
        self.test_runner = test_runner or TestRunner(
            command=self.settings.test_command,
            timeout=self.settings.test_timeout_seconds,
        )
        self.conflict_resolver = conflict_resolver or ConflictResolver(llm=llm)
        self.reviewer = reviewer or FinalReviewer(llm=llm)
        self.event_bus = event_bus or get_event_bus()
        self.orchestration_id = orchestration_id
        self.collector = ChangeCollector()
        self.regression = RegressionDetector()

    # ── main entry point ────────────────────────────────────────────────────

    async def integrate(self, state: ExecutionState) -> IntegrationReport:
        started = time.time()
        await self._emit(
            EventType.INTEGRATION_STARTED, repository=state.repository
        )

        report = IntegrationReport()
        target = await self.git.default_branch()
        baseline_sha = await self.git.snapshot_ref()
        await self.git.ensure_identity()

        # Baseline: what did the suite do *before* we merged anything?
        baseline = await self._run_tests(state)

        mergeable, skipped = await self.collector.collect(state, self.git)
        report.skipped_branches = [c.branch or c.task_id for c in skipped]
        report.skipped_details = [c.to_dict() for c in skipped]
        logger.info(
            "Integration starting",
            extra={"mergeable": len(mergeable), "skipped": len(skipped), "target": target},
        )

        # A run where agents succeeded but produced nothing mergeable is not a
        # success. Surface it rather than letting a green-looking report hide it.
        #
        # Read-only task kinds are exempt: an analyst and a reviewer that commit
        # nothing did their job. Counting them made every well-formed plan look
        # incomplete, because the analysis and review waves always end this way.
        successful_without_branch = [
            c
            for c in skipped
            if c.status == TaskStatus.SUCCESS.value
            and c.skip_reason
            and c.task_type not in _READ_ONLY_TASK_TYPES
        ]
        if successful_without_branch and not mergeable:
            for record in successful_without_branch:
                state.record_error(
                    f"Task '{record.title}' reported success but {record.skip_reason}"
                )

        for record in mergeable:
            outcome = await self._merge_one(record, report, target)
            # `outcome` is also returned for a branch that was left unmerged, so
            # success -- not mere existence -- is what earns a place here.
            if outcome is not None and outcome.success:
                report.merged_branches.append(record.branch or record.task_id)

        # Verify the integrated state.
        after = await self._run_tests(state)
        report.test_runs = [r for r in (baseline, after) if r is not None]
        state.test_results = {
            "baseline_passed": baseline.passed if baseline else None,
            "after_passed": after.passed,
            "passed_count": after.passed_count,
            "failed_count": after.failed_count,
            "command": after.command,
            "failures": _extract_failures(after),
            "summary": after.summary,
        }
        await self._emit(
            EventType.TESTS_FINISHED,
            passed=after.passed,
            passed_count=after.passed_count,
            failed_count=after.failed_count,
            command=after.command,
        )

        regressions = self.regression.compare(baseline, after)
        report.regressions = regressions

        if not after.passed and regressions:
            # The merge broke something. Roll back rather than leave the
            # repository in a state that fails its own tests.
            logger.warning("Rolling back failed integration", extra={"regressions": regressions})
            for regression in regressions:
                state.record_error(regression)
            await self.git.revert_to(baseline_sha)
            report.success = False
            report.report = self._render(report, state, target, baseline_sha)
            state.final_status = FinalStatus.FAILED
            await self._emit(
                EventType.INTEGRATION_FINISHED, success=False, regressions=regressions
            )
            return report

        # Diff for review, relative to the pre-merge snapshot.
        diff = await self.git.diff(baseline_sha, target, stat_only=False)
        critical_findings: list[str] = []
        if self.llm_available():
            report.review_findings = await self.reviewer.review(diff.patch)
            for finding in report.review_findings:
                if finding.get("severity") == "critical":
                    message = f"Critical review finding: {finding.get('issue')}"
                    critical_findings.append(message)
                    regressions.append(message)

        # Clean up. The worktrees go first: git refuses to delete a branch that
        # is still checked out in a worktree, so the other order leaves every
        # agent branch behind.
        await self._prune_worktrees()
        # Merged branches, plus the no-op branches of tasks that succeeded with
        # nothing to merge -- those hold no evidence worth keeping. Branches of
        # *failed* tasks are left alone: the executor keeps those workspaces on
        # purpose so the failure can be inspected afterwards.
        stale = [r for r in mergeable if r.branch] + [
            r for r in skipped if r.branch and r.status == TaskStatus.SUCCESS.value
        ]
        for record in stale:
            await self.git.delete_branch(record.branch)

        report.regressions = regressions
        unresolved = [
            path
            for resolution in report.conflict_resolutions
            for path in resolution.get("unresolved", [])
        ]
        if unresolved:
            for path in unresolved:
                state.record_error(f"Merge conflict could not be resolved: {path}")
        # A run that merged nothing is not a success, even when the (untouched)
        # suite passes. This is the "9 tasks succeeded but 0 branches were
        # merged" case the report exists to make visible. A critical review
        # finding downgrades the run too -- otherwise the reviewer is advisory
        # in name only.
        report.success = bool(
            after.passed
            and not unresolved
            and not successful_without_branch
            and not critical_findings
            and report.merged_branches
        )
        state.final_status = (
            FinalStatus.SUCCESS
            if report.success
            else FinalStatus.PARTIAL_SUCCESS
            if report.merged_branches
            else FinalStatus.FAILED
        )
        state.integration_report = report
        state.touch()
        report.report = self._render(report, state, target, baseline_sha)
        await self._emit(
            EventType.INTEGRATION_FINISHED,
            success=report.success,
            merged=len(report.merged_branches),
            conflicts=len(report.conflicts),
        )
        return report

    # ── merging ─────────────────────────────────────────────────────────────

    async def _merge_one(
        self, record: ChangeRecord, report: IntegrationReport, target: str
    ) -> Optional[MergeOutcome]:
        branch = record.branch
        if not branch:
            return None
        await self._emit(
            EventType.AGENT_FINISHED,
            phase="integration",
            branch=branch,
            task_id=record.task_id,
            title=record.title,
        )
        outcome = await self.git.merge(branch, target)
        if outcome.success:
            logger.info("Merged branch", extra={"branch": branch, "task": record.title})
            return outcome

        # Conflict: resolve it properly or leave the branch unmerged.
        if not outcome.conflicts:
            state_note = outcome.message or "merge failed without reported conflicts"
            logger.warning("Merge failed", extra={"branch": branch, "reason": state_note})
            return outcome

        await self._emit(
            EventType.CONFLICT_DETECTED,
            branch=branch,
            paths=outcome.conflicts,
            task_id=record.task_id,
        )
        report.conflicts.append(
            {
                "branch": branch,
                "paths": outcome.conflicts,
                "task_id": record.task_id,
                "title": record.title,
                "details": [
                    {k: v for k, v in d.items() if k != "raw"}
                    for d in outcome.conflict_details
                ],
            }
        )

        resolution = await self.conflict_resolver.resolve_with_report(
            str(self.git.repo_path), outcome.conflict_details
        )
        report.conflict_resolutions.append(
            {"branch": branch, **resolution.to_dict()}
        )
        await self._emit(
            EventType.CONFLICT_RESOLVED,
            branch=branch,
            resolved=resolution.resolved,
            unresolved=resolution.unresolved,
            strategies=resolution.strategies,
        )
        if not resolution.success:
            logger.error(
                "Could not resolve every conflict; leaving branch unmerged",
                extra={"branch": branch, "unresolved": resolution.unresolved},
            )
            return outcome

        merged = await self.git.merge_and_resolve(
            branch, resolution.resolved_contents, target
        )
        if merged.success:
            logger.info("Merged with resolution", extra={"branch": branch})
        return merged

    # ── helpers ─────────────────────────────────────────────────────────────

    async def _run_tests(self, state: ExecutionState) -> TestRunResult:
        if not self.test_runner.can_run(self.git.repo_path):
            logger.info("No test command detected; skipping verification")
            return TestRunResult(
                command="", passed=True, skipped=True,
                stdout="No test command detected in this repository.",
            )
        await self._emit(EventType.TESTS_STARTED, repo=self.git.repo_path)
        return await self.test_runner.run(self.git.repo_path)

    async def _prune_worktrees(self) -> None:
        for worktree in await self.git.list_worktrees():
            if worktree.get("branch", "").startswith("agent/"):
                await self.git.remove_worktree(worktree["path"])

    def llm_available(self) -> bool:
        return self.conflict_resolver.llm is not None

    def _render(
        self,
        report: IntegrationReport,
        state: ExecutionState,
        target: str,
        baseline_sha: str,
    ) -> str:
        lines: list[str] = []
        lines.append(f"# Integration report")
        lines.append("")
        lines.append(f"- Target branch: `{target}`")
        lines.append(f"- Baseline commit: `{baseline_sha[:10]}`")
        lines.append(f"- Merged branches: {len(report.merged_branches)}")
        lines.append(f"- Skipped tasks: {len(report.skipped_branches)}")
        lines.append(f"- Conflicts encountered: {len(report.conflicts)}")
        lines.append(f"- Outcome: {'SUCCESS' if report.success else 'INCOMPLETE'}")
        if report.merged_branches:
            lines.append("")
            lines.append("## Merged")
            for branch in report.merged_branches:
                lines.append(f"- `{branch}`")
        if report.conflicts:
            lines.append("")
            lines.append("## Conflicts")
            for conflict in report.conflicts:
                paths = ", ".join(conflict.get("paths", []))
                lines.append(f"- `{conflict.get('branch')}`: {paths}")
        for resolution in report.conflict_resolutions:
            strategies = resolution.get("strategies", {})
            if strategies:
                lines.append("")
                lines.append("## Resolution strategies")
                for path, how in strategies.items():
                    lines.append(f"- `{path}` -> {how}")
            if resolution.get("unresolved"):
                lines.append("")
                lines.append("## Unresolved")
                for path in resolution["unresolved"]:
                    lines.append(f"- `{path}`")
        if report.test_runs:
            lines.append("")
            lines.append("## Tests")
            for run in report.test_runs:
                if run.skipped:
                    lines.append(f"- {run.stdout}")
                else:
                    lines.append(
                        f"- `{run.command}` -> {'PASS' if run.passed else 'FAIL'} "
                        f"({run.passed_count} passed, {run.failed_count} failed, "
                        f"{run.duration_seconds}s)"
                    )
        if report.regressions:
            lines.append("")
            lines.append("## Regressions")
            lines.extend(f"- {r}" for r in report.regressions)
        if report.review_findings:
            lines.append("")
            lines.append("## Review findings")
            for finding in report.review_findings:
                location = finding.get("file", "")
                if location:
                    location = f" `{location}`"
                lines.append(
                    f"- [{finding.get('severity', 'info')}]{location} {finding.get('issue', '')}"
                )
        return "\n".join(lines)

    async def _emit(self, event_type: EventType, **data) -> None:
        if not self.orchestration_id:
            return
        import contextlib

        with contextlib.suppress(Exception):
            await self.event_bus.publish(self.orchestration_id, event_type, **data)


def _extract_failures(result: TestRunResult) -> list[str]:
    """Pull the failing test names out of a pytest-style report."""
    import re

    if result.passed or not result.stdout:
        return []
    failures = re.findall(r"^FAILED\s+(\S+)", result.stdout, re.M)
    if not failures:
        failures = re.findall(r"^(?:FAIL|✗)\s+(\S+)", result.stdout, re.M)
    if not failures:
        errors = re.findall(r"^E\s+(.{20,200})$", result.stdout, re.M)
        failures = errors[:10]
    return failures[:25]

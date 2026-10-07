"""
GitManager.

All git access goes through one place, asynchronously, via `asyncio` subprocess
calls so a slow `git merge` never blocks the event loop mid-orchestration.

The important part is worktree lifecycle: every agent task gets its own
worktree on its own branch, and branches are only deleted once the integrator
has merged them. That is the mechanism which makes parallel writes safe --
without it, two agents editing the same directory is data loss.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from app.logging_config import get_logger

logger = get_logger("app.git")

DEFAULT_TIMEOUT = 120.0


class GitError(RuntimeError):
    """A git command failed."""

    def __init__(self, command: Sequence[str], returncode: int, stderr: str):
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"git {' '.join(self.command)} failed ({returncode}): {stderr.strip()[:400]}"
        )


@dataclass
class CommitInfo:
    sha: str
    message: str
    author: str = ""
    files: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0


@dataclass
class DiffStat:
    files_changed: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    patch: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.files_changed


@dataclass
class MergeOutcome:
    success: bool
    conflicts: list[str] = field(default_factory=list)
    conflict_details: list[dict] = field(default_factory=list)
    commit: Optional[str] = None
    message: str = ""
    already_merged: bool = False


class GitManager:
    """Async wrapper around the git CLI for one repository."""

    def __init__(self, repo_path: str | Path, timeout: float = DEFAULT_TIMEOUT):
        self.repo_path = str(Path(repo_path).expanduser().resolve())
        self.timeout = timeout

    # ── low-level ───────────────────────────────────────────────────────────

    async def run(
        self,
        *args: str,
        cwd: Optional[str] = None,
        check: bool = True,
        timeout: Optional[float] = None,
    ) -> str:
        """Run a git command. Raises GitError when `check` and it fails."""
        command = ["git", *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd or self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise GitError(command, 127, "git executable not found on PATH") from None
        except OSError as exc:
            # A missing or unreadable working directory raises here; without this
            # the caller sees an opaque OSError instead of a git problem.
            raise GitError(command, 126, f"cannot run git in {cwd or self.repo_path}: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout or self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise GitError(command, 124, f"timed out after {timeout or self.timeout}s")

        out = stdout.decode(errors="replace").rstrip()
        err = stderr.decode(errors="replace").rstrip()
        if check and proc.returncode != 0:
            raise GitError(command, proc.returncode or 1, err or out)
        if err and proc.returncode != 0:
            logger.debug("git stderr", extra={"cmd": " ".join(args), "stderr": err[:300]})
        return out

    async def run_result(self, *args: str, cwd: Optional[str] = None) -> tuple[int, str, str]:
        """Run git and return (returncode, stdout, stderr) without raising."""
        command = ["git", *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd or self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except FileNotFoundError:
            return 127, "", "git executable not found on PATH"
        except OSError as exc:
            return 126, "", f"cannot run git in {cwd or self.repo_path}: {exc}"
        except asyncio.TimeoutError:
            return 124, "", "timeout"
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace").rstrip(),
            stderr.decode(errors="replace").rstrip(),
        )

    # ── repository lifecycle ────────────────────────────────────────────────

    async def is_repo(self) -> bool:
        # Probing a directory that does not exist raises on some platforms,
        # so the check is guarded rather than relying on git's exit code.
        if not Path(self.repo_path).is_dir():
            return False
        code, out, _ = await self.run_result("rev-parse", "--is-inside-work-tree")
        return code == 0 and out == "true"

    async def init_repo(self, initial_branch: str = "main") -> None:
        """
        Create a repository if one is not already there.

        Always produces a root commit, even when the directory already has
        content: `git worktree add` needs a HEAD to branch from, so an
        "initialised but empty history" repo would fail every agent task.
        """
        Path(self.repo_path).mkdir(parents=True, exist_ok=True)
        if await self.is_repo():
            if await self.has_commits():
                return
            await self.ensure_identity()
        else:
            await self.run("init", "--initial-branch", initial_branch)
            await self.ensure_identity()

        gitignore = Path(self.repo_path) / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                "__pycache__/\n*.pyc\n.pytest_cache/\n.env\nnode_modules/\n",
                encoding="utf-8",
            )
        await self.run("add", "-A")
        await self.run("commit", "-m", "Initial commit")

    async def has_commits(self) -> bool:
        code, _, _ = await self.run_result("rev-parse", "--verify", "HEAD")
        return code == 0

    async def ensure_identity(self) -> None:
        """Guarantee a committer identity so commits cannot fail on a fresh box."""
        code, out, _ = await self.run_result("config", "user.email")
        if code != 0 or not out:
            await self.run("config", "user.email", "agents@orchestrator.local")
            await self.run("config", "user.name", "Orchestrator Agent")

    async def current_branch(self) -> str:
        return await self.run("rev-parse", "--abbrev-ref", "HEAD")

    async def head_commit(self, cwd: Optional[str] = None) -> str:
        return await self.run("rev-parse", "HEAD", cwd=cwd)

    async def default_branch(self) -> str:
        """
        The branch an integration target should be.

        Never hard-coded: honours the current checkout, then `origin/HEAD`,
        then falls back to whatever branch exists. A detached HEAD (which CI
        checkouts and `worktree add --detach` both produce) is *not* a branch
        name, so it falls through to the next strategy.
        """
        code, out, _ = await self.run_result("rev-parse", "--abbrev-ref", "HEAD")
        if code == 0 and out and out != "HEAD" and not out.startswith("(HEAD "):
            return out
        code, out, _ = await self.run_result("symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        if code == 0 and out:
            return out.rsplit("/", 1)[-1]
        code, out, _ = await self.run_result("branch", "--format=%(refname:short)")
        for line in out.splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("agent/"):
                continue
            # `git branch` lists a detached HEAD as a pseudo-entry such as
            # "(HEAD detached at 967b93b)", which is not a branch name.
            if candidate.startswith("(") or candidate.startswith("*"):
                continue
            return candidate
        return "main"

    # ── status & inspection ─────────────────────────────────────────────────

    async def status(self, cwd: Optional[str] = None) -> list[dict[str, str]]:
        """
        Porcelain status as a list of {path, status, unmerged} dicts.

        Uses the plain (non `-z`) porcelain=v1 form: with `-z` the XY prefix is
        dropped, so the `UU` conflict marker never appears in the output. The
        XY prefix is meaningful even when it is a single space, so only the
        trailing newline is stripped -- a leading `strip()` here would silently
        truncate the path of the first entry.
        """
        out = await self.run("status", "--porcelain=v1", cwd=cwd, check=False)
        entries: list[dict[str, str]] = []
        for line in out.splitlines():
            if len(line) < 4:
                continue
            code, path = line[:2], line[3:].strip()
            entries.append(
                {
                    "path": path,
                    "status": code,
                    "unmerged": code in ("UU", "AA", "DD", "AU", "UA", "DU", "UD"),
                }
            )
        return entries

    async def is_dirty(self, cwd: Optional[str] = None) -> bool:
        return bool(await self.status(cwd=cwd))

    async def list_branches(self) -> list[str]:
        out = await self.run("branch", "--format=%(refname:short)", check=False)
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def list_worktrees(self) -> list[dict[str, str]]:
        out = await self.run("worktree", "list", "--porcelain", check=False)
        worktrees: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in out.splitlines():
            if not line.strip():
                if current:
                    worktrees.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            if key == "worktree":
                if current:
                    worktrees.append(current)
                current = {"path": value}
            elif key == "branch":
                current["branch"] = value.replace("refs/heads/", "")
            elif key == "HEAD":
                current["head"] = value
        if current:
            worktrees.append(current)
        return worktrees

    async def branch_exists(self, branch: str) -> bool:
        code, _, _ = await self.run_result("rev-parse", "--verify", f"refs/heads/{branch}")
        return code == 0

    # ── worktrees ───────────────────────────────────────────────────────────

    async def create_worktree(
        self,
        branch: str,
        path: str | Path,
        base_point: str = "HEAD",
    ) -> Path:
        """
        Create an isolated checkout on its own branch.

        This is the mechanism that stops two agents writing the same files.
        """
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if await self.branch_exists(branch):
            await self.run("worktree", "add", str(target), branch)
        else:
            await self.run("worktree", "add", "-b", branch, str(target), base_point)
        return target

    async def remove_worktree(self, path: str | Path, force: bool = True) -> bool:
        """Detach and delete a worktree directory."""
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(Path(path).resolve()))
        code, _, _ = await self.run_result(*args)
        if code != 0:
            # Fall back to a manual cleanup for worktrees git has lost track of.
            if Path(path).exists():
                shutil.rmtree(path, ignore_errors=True)
                await self.run_result("worktree", "prune")
        return not Path(path).exists()

    async def delete_branch(self, branch: str, force: bool = True) -> bool:
        args = ["branch", "-D" if force else "-d", branch]
        code, _, _ = await self.run_result(*args)
        return code == 0

    # ── commits ─────────────────────────────────────────────────────────────

    async def commit_all(self, cwd: str | Path, message: str) -> Optional[str]:
        """Stage everything and commit. Returns None when there is nothing to commit."""
        await self.run("add", "-A", cwd=str(cwd))
        code, out, _ = await self.run_result("diff", "--cached", "--quiet", cwd=str(cwd))
        if code == 0:
            return None  # nothing staged
        await self.run("commit", "-m", message, cwd=str(cwd))
        return await self.head_commit(cwd=str(cwd))

    async def commit_info(self, ref: str = "HEAD") -> CommitInfo:
        fmt = "%H%x00%s%x00%an"
        out = await self.run("show", "-s", f"--format={fmt}", ref, check=False)
        parts = out.split("\0")
        sha = parts[0] if parts else ""
        message = parts[1] if len(parts) > 1 else ""
        author = parts[2] if len(parts) > 2 else ""
        files: list[str] = []
        stat_out = await self.run("show", "--name-only", "--format=", ref, check=False)
        files = [line.strip() for line in stat_out.splitlines() if line.strip()]
        return CommitInfo(sha=sha, message=message, author=author, files=files)

    # ── diffs ───────────────────────────────────────────────────────────────

    async def diff(
        self,
        ref_a: str,
        ref_b: str = "HEAD",
        cwd: Optional[str] = None,
        stat_only: bool = False,
    ) -> DiffStat:
        args = ["diff", ref_a, ref_b]
        if stat_only:
            args.append("--stat")
        patch = await self.run(*args, cwd=cwd, check=False)
        name_out = await self.run(
            "diff", "--name-only", ref_a, ref_b, cwd=cwd, check=False
        )
        files = [line.strip() for line in name_out.splitlines() if line.strip()]
        insertions, deletions = 0, 0
        shortstat = await self.run(
            "diff", "--shortstat", ref_a, ref_b, cwd=cwd, check=False
        )
        ins_match = re.search(r"(\d+) insertions?", shortstat)
        del_match = re.search(r"(\d+) deletions?", shortstat)
        if ins_match:
            insertions = int(ins_match.group(1))
        if del_match:
            deletions = int(del_match.group(1))
        return DiffStat(
            files_changed=files,
            insertions=insertions,
            deletions=deletions,
            patch="" if stat_only else patch,
        )

    async def diff_working_tree(self, cwd: Optional[str] = None) -> DiffStat:
        """
        Uncommitted changes (staged and unstaged) relative to HEAD.

        Note the single ref: `git diff HEAD HEAD` compares a commit with itself
        and is therefore always empty, which would silently report "no changes".
        """
        out = await self.run("diff", "HEAD", cwd=cwd, check=False)
        name_out = await self.run("diff", "HEAD", "--name-only", cwd=cwd, check=False)
        files = [line.strip() for line in name_out.splitlines() if line.strip()]
        shortstat = await self.run(
            "diff", "HEAD", "--shortstat", cwd=cwd, check=False
        )
        ins = re.search(r"(\d+) insertions?", shortstat)
        dele = re.search(r"(\d+) deletions?", shortstat)
        return DiffStat(
            files_changed=files,
            insertions=int(ins.group(1)) if ins else 0,
            deletions=int(dele.group(1)) if dele else 0,
            patch=out,
        )

    async def show_file(self, ref: str, path: str) -> str:
        return await self.run("show", f"{ref}:{path}", check=False)

    async def file_at(self, ref: str, path: str) -> Optional[str]:
        code, out, _ = await self.run_result("show", f"{ref}:{path}")
        return out if code == 0 else None

    # ── merging & conflicts ─────────────────────────────────────────────────

    async def detect_conflicts(
        self, source_branch: str, target_branch: str
    ) -> list[dict]:
        """
        Preview whether a merge would conflict, without touching the tree.

        `git merge-tree --write-tree` performs a real merge in memory and exits
        non-zero on conflict. Line 1 is the resulting tree OID; subsequent lines
        of the form `<mode> <oid> <stage>\\t<path>` describe each conflict
        stage, followed by a blank line and human-readable messages.
        """
        code, out, err = await self.run_result(
            "merge-tree", "--write-tree", source_branch, target_branch
        )
        if code == 0:
            return []
        if code == 129 or not out.strip():
            logger.warning(
                "merge-tree preview unavailable",
                extra={"source": source_branch, "error": (err or out)[:200]},
            )
            return []
        return self._parse_merge_tree_conflicts(out)

    @staticmethod
    def _parse_merge_tree_conflicts(output: str) -> list[dict]:
        """Group merge-tree stage lines into one record per conflicted path."""
        by_path: dict[str, dict] = {}
        messages: list[str] = []
        for line in output.splitlines():
            if not line.strip():
                messages.append("")
                continue
            match = re.match(r"^(\d{6})\s+([0-9a-f]{40})\s+([123])\t(.+)$", line)
            if match:
                mode, oid, stage, path = match.groups()
                entry = by_path.setdefault(
                    path, {"path": path, "stages": 0, "conflict_type": "content"}
                )
                entry["stages"] += 1
                entry[f"stage{stage}"] = oid
                continue
            if re.match(r"^[0-9a-f]{40}$", line.strip()):
                continue  # the tree OID on line 1
            messages.append(line.strip())
        for entry in by_path.values():
            entry["message"] = " ".join(m for m in messages if m)[:300]
        return list(by_path.values())

    async def merge(
        self,
        source_branch: str,
        target_branch: Optional[str] = None,
        message: Optional[str] = None,
    ) -> MergeOutcome:
        """
        Merge `source_branch` into `target_branch`.

        On conflict the merge is aborted, the working tree is left clean, and
        the conflicting paths are reported so a resolver agent can handle them.
        Use `merge_and_resolve` when a resolution is ready.
        """
        target = target_branch or await self.default_branch()
        if not await self.branch_exists(source_branch):
            return MergeOutcome(success=False, message=f"branch {source_branch} does not exist")

        current = await self.current_branch()
        if current != target:
            await self.run("checkout", target, check=False)

        # Already contained in the target? Nothing to do.
        code, _, _ = await self.run_result(
            "merge-base", "--is-ancestor", source_branch, target
        )
        if code == 0:
            return MergeOutcome(
                success=True,
                already_merged=True,
                message=f"{source_branch} is already contained in {target}",
            )

        merge_message = message or f"Merge branch '{source_branch}' into {target}"
        code, out, err = await self.run_result(
            "merge", "--no-ff", "--no-edit", "-m", merge_message, source_branch
        )
        if code == 0:
            return MergeOutcome(
                success=True,
                commit=await self.head_commit(),
                message=out or merge_message,
            )

        conflicts = await self._collect_unmerged()
        details = await self._conflict_details(conflicts)
        await self.abort_merge()
        return MergeOutcome(
            success=False,
            conflicts=conflicts,
            conflict_details=details,
            message=err or out or "merge failed",
        )

    async def begin_merge(
        self,
        source_branch: str,
        target_branch: Optional[str] = None,
        message: Optional[str] = None,
    ) -> MergeOutcome:
        """
        Start a merge and LEAVE it in the conflicted state.

        The caller is expected to call `write_resolution` for every reported
        path and then `commit_merge`, which completes the merge properly (the
        resulting commit records both parents, so the branch is genuinely
        integrated). Always pair with `abort_merge` on the failure path.

        Uncommitted changes to tracked files on the target are discarded first:
        git refuses to merge over them, and the resolutions are supplied by the
        caller anyway, so keeping them would only abort the merge and throw
        the resolved content away.
        """
        target = target_branch or await self.default_branch()
        if not await self.branch_exists(source_branch):
            return MergeOutcome(success=False, message=f"branch {source_branch} does not exist")
        current = await self.current_branch()
        if current != target:
            await self.run("checkout", target, check=False)
        if await self.is_dirty():
            logger.info(
                "Discarding uncommitted target changes before merging",
                extra={"target": target},
            )
            await self.run("checkout", "--", ".", check=False)
        code, _, _ = await self.run_result("merge-base", "--is-ancestor", source_branch, target)
        if code == 0:
            return MergeOutcome(
                success=True,
                already_merged=True,
                message=f"{source_branch} is already contained in {target}",
            )
        merge_message = message or f"Merge branch '{source_branch}' into {target}"
        code, out, err = await self.run_result(
            "merge", "--no-ff", "--no-edit", "-m", merge_message, source_branch
        )
        if code == 0:
            return MergeOutcome(success=True, commit=await self.head_commit(), message=merge_message)
        conflicts = await self._collect_unmerged()
        details = await self._conflict_details(conflicts)
        return MergeOutcome(
            success=False,
            conflicts=conflicts,
            conflict_details=details,
            message=err or out or "merge failed",
        )

    async def merge_and_resolve(
        self,
        source_branch: str,
        resolutions: dict[str, str],
        target_branch: Optional[str] = None,
        message: Optional[str] = None,
    ) -> MergeOutcome:
        """
        Merge, write the supplied resolution for every conflict, and commit.

        On any failure the merge is aborted so the target branch is never left
        in a half-merged state.
        """
        outcome = await self.begin_merge(source_branch, target_branch, message)
        if outcome.success:
            return outcome
        if not outcome.conflicts:
            await self.abort_merge()
            return outcome

        missing = [p for p in outcome.conflicts if p not in resolutions]
        if missing:
            await self.abort_merge()
            return MergeOutcome(
                success=False,
                conflicts=outcome.conflicts,
                conflict_details=outcome.conflict_details,
                message=f"no resolution supplied for: {', '.join(missing)}",
            )

        try:
            await self.write_resolution(
                {p: resolutions[p] for p in outcome.conflicts}
            )
            commit = await self.commit_merge(
                message or f"Merge branch '{source_branch}' (conflicts resolved)"
            )
        except Exception as exc:  # noqa: BLE001
            await self.abort_merge()
            return MergeOutcome(
                success=False,
                conflicts=outcome.conflicts,
                conflict_details=outcome.conflict_details,
                message=f"resolution failed: {exc}",
            )
        return MergeOutcome(
            success=True,
            conflicts=outcome.conflicts,
            conflict_details=outcome.conflict_details,
            commit=commit,
            message="merge completed with resolved conflicts",
        )

    async def _collect_unmerged(self) -> list[str]:
        out = await self.run("diff", "--name-only", "--diff-filter=U", check=False)
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def _conflict_details(self, paths: list[str]) -> list[dict]:
        details: list[dict] = []
        for path in paths:
            try:
                content = Path(self.repo_path, path).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                content = ""
            sections = self._split_conflict_markers(content)
            details.append(
                {
                    "path": path,
                    "conflict_count": content.count("<<<<<<<"),
                    "ours": sections[0],
                    "theirs": sections[1],
                    "raw": content[:8000],
                }
            )
        return details

    @staticmethod
    def _split_conflict_markers(content: str) -> tuple[str, str]:
        """Extract the ours/theirs halves of a conflicted file."""
        ours: list[str] = []
        theirs: list[str] = []
        side = None
        for line in content.splitlines():
            if line.startswith("<<<<<<<"):
                side = "ours"
                continue
            if line.startswith("|||||||"):
                side = "base"
                continue
            if line.startswith("======="):
                side = "theirs"
                continue
            if line.startswith(">>>>>>>"):
                side = None
                continue
            if side == "ours":
                ours.append(line)
            elif side == "theirs":
                theirs.append(line)
        return "\n".join(ours).strip(), "\n".join(theirs).strip()

    async def write_resolution(self, resolutions: dict[str, str]) -> list[str]:
        """Write merged contents and stage them, leaving the merge ready to commit."""
        written: list[str] = []
        for path, content in resolutions.items():
            full = Path(self.repo_path, path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            written.append(path)
        if written:
            await self.run("add", "--", *written)
        return written

    async def commit_merge(self, message: str) -> Optional[str]:
        return await self.commit_all(self.repo_path, message)

    async def abort_merge(self) -> None:
        await self.run("merge", "--abort", check=False)
        await self.run("reset", "--hard", check=False)

    async def reset_hard(self, ref: Optional[str] = None) -> None:
        # `ref` is optional, so it must not be passed to exec when unset.
        args = ["reset", "--hard"] + ([ref] if ref else [])
        await self.run(*args, check=False)
        await self.run("clean", "-fd", check=False)

    async def revert_to(self, ref: str) -> None:
        """Return the repo to a known-good state after a failed integration."""
        await self.run("reset", "--hard", ref, check=False)
        await self.run("clean", "-fd", check=False)

    # ── sync helpers ────────────────────────────────────────────────────────

    async def fetch(self, remote: str = "origin") -> bool:
        code, _, _ = await self.run_result("fetch", remote, "--prune")
        return code == 0

    async def snapshot_ref(self) -> str:
        """A commit the integrator can roll back to."""
        return await self.head_commit()

    def worktree_path_for(self, root: str | Path, task_id: str) -> Path:
        return Path(root) / task_id


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (cleaned[:40] or "task").rstrip("-")


def is_git_installed() -> bool:
    return shutil.which("git") is not None


def repo_size_hint(repo_path: str | Path) -> str:
    """Cheap 'is this a real project' check used by the repository analyzer."""
    root = Path(repo_path)
    if not root.exists():
        return "missing"
    for marker in (".git", "pyproject.toml", "package.json", "go.mod", "Cargo.toml"):
        if (root / marker).exists():
            return "present"
    return "empty"

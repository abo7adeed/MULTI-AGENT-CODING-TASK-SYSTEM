"""
GitManager tests.

These drive a real `git` binary against throwaway repositories in tmp. The
worktree / merge / conflict behaviour is the part of the system where a silent
mistake destroys an agent's work, so it is tested against real git rather than
a fake.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from app.git.manager import (
    CommitInfo,
    DiffStat,
    GitError,
    GitManager,
    MergeOutcome,
    _slugify,
    is_git_installed,
    repo_size_hint,
)

pytestmark = pytest.mark.skipif(
    not is_git_installed(), reason="git is not installed on PATH"
)

# Blob OIDs as `git merge-tree --write-tree` prints them: 40 lowercase hex chars.
OID_BASE = "a" * 40
OID_OURS = "b" * 40
OID_THEIRS = "c" * 40
TREE_OID = "1" * 40


def run(cwd: Path, *args: str, check: bool = True) -> str:
    """Synchronous helper for arranging repository state in a test."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    ).stdout.strip()


def write(path: Path, name: str, content: str) -> Path:
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


class TestLowLevel:
    @pytest.mark.asyncio
    async def test_run_returns_stdout(self, git):
        assert await git.run("rev-parse", "--is-inside-work-tree") == "true"

    @pytest.mark.asyncio
    async def test_checked_failure_raises_with_context(self, git):
        with pytest.raises(GitError) as excinfo:
            await git.run("rev-parse", "--verify", "no-such-ref")
        error = excinfo.value
        assert error.returncode != 0
        assert "rev-parse" in error.command
        assert error.stderr

    @pytest.mark.asyncio
    async def test_unchecked_failure_returns_output_instead(self, git):
        assert await git.run("rev-parse", "--verify", "no-such-ref", check=False) == ""

    @pytest.mark.asyncio
    async def test_run_result_never_raises(self, git):
        code, _, _ = await git.run_result("rev-parse", "--verify", "no-such-ref")
        assert code != 0

    @pytest.mark.asyncio
    async def test_run_result_reports_a_bad_flag_without_raising(self, tmp_path):
        manager = GitManager(tmp_path)
        code, _, err = await manager.run_result("--this-is-not-a-git-flag")
        assert code != 0
        assert err

    @pytest.mark.asyncio
    async def test_timeout_kills_the_process(self, tmp_path):
        manager = GitManager(tmp_path, timeout=0.05)
        with pytest.raises(GitError) as excinfo:
            # `--exec-path` is instant; sleep forces the timeout path.
            await manager.run(
                "-c", "alias.zzz=!sleep 5", "zzz", timeout=0.05
            )
        assert excinfo.value.returncode == 124


class TestRepoLifecycle:
    @pytest.mark.asyncio
    async def test_is_repo_is_false_for_a_plain_directory(self, repo_path):
        assert await GitManager(repo_path).is_repo() is False

    @pytest.mark.asyncio
    async def test_is_repo_is_false_for_a_missing_directory(self, tmp_path):
        assert await GitManager(tmp_path / "nope").is_repo() is False

    @pytest.mark.asyncio
    async def test_is_repo_is_false_for_a_subdirectory_of_a_repo(self, seed_repo, git):
        # A subdirectory of a repo is still inside a work tree, so this is true.
        assert await GitManager(seed_repo / "tests").is_repo() is True

    @pytest.mark.asyncio
    async def test_init_creates_a_branch_and_a_root_commit(self, repo_path):
        manager = GitManager(repo_path)
        await manager.init_repo()
        assert await manager.current_branch() == "main"
        assert await manager.has_commits() is True

    @pytest.mark.asyncio
    async def test_init_over_a_populated_directory_still_commits(
        self, repo_path
    ):
        """Without a root commit there is no HEAD and worktree creation fails."""
        write(repo_path, "main.py", "print('hi')\n")
        manager = GitManager(repo_path)
        await manager.init_repo()
        assert await manager.has_commits() is True
        assert (repo_path / ".gitignore").exists()

    @pytest.mark.asyncio
    async def test_init_uses_the_requested_branch_name(self, repo_path):
        manager = GitManager(repo_path)
        await manager.init_repo(initial_branch="trunk")
        assert await manager.current_branch() == "trunk"

    @pytest.mark.asyncio
    async def test_init_is_idempotent(self, repo_path):
        manager = GitManager(repo_path)
        await manager.init_repo()
        first = await manager.head_commit()
        await manager.init_repo()
        assert await manager.head_commit() == first

    @pytest.mark.asyncio
    async def test_init_does_not_overwrite_an_existing_gitignore(self, repo_path):
        (repo_path / ".gitignore").write_text("custom/\n", encoding="utf-8")
        manager = GitManager(repo_path)
        await manager.init_repo()
        assert (repo_path / ".gitignore").read_text(encoding="utf-8") == "custom/\n"

    @pytest.mark.asyncio
    async def test_identity_is_configured_when_absent(self, repo_path):
        manager = GitManager(repo_path)
        run(repo_path, "init", "--initial-branch", "main")
        # Blank rather than unset: a global identity would otherwise answer.
        run(repo_path, "config", "user.email", "")
        await manager.ensure_identity()
        assert await manager.run("config", "user.email") == "agents@orchestrator.local"

    @pytest.mark.asyncio
    async def test_existing_identity_is_left_alone(self, repo_path):
        manager = GitManager(repo_path)
        run(repo_path, "init", "--initial-branch", "main")
        run(repo_path, "config", "user.email", "real@developer.dev")
        await manager.ensure_identity()
        assert await manager.run("config", "user.email") == "real@developer.dev"

    @pytest.mark.asyncio
    async def test_default_branch_follows_the_checkout(self, repo_path):
        manager = GitManager(repo_path)
        await manager.init_repo()
        run(repo_path, "checkout", "-b", "release")
        assert await manager.default_branch() == "release"

    @pytest.mark.asyncio
    async def test_default_branch_skips_agent_branches_when_detached(self, repo_path):
        manager = GitManager(repo_path)
        await manager.init_repo()
        run(repo_path, "branch", "agent/one")
        run(repo_path, "checkout", "--detach", "HEAD")
        branch = await manager.default_branch()
        assert branch == "main"
        assert await manager.branch_exists(branch) is True

    @pytest.mark.asyncio
    async def test_branch_exists(self, git):
        assert await git.branch_exists("main") is True
        assert await git.branch_exists("nope") is False


class TestStatus:
    @pytest.mark.asyncio
    async def test_clean_tree_is_empty(self, git):
        assert await git.status() == []
        assert await git.is_dirty() is False

    @pytest.mark.asyncio
    async def test_modified_untracked_and_deleted_are_classified(self, git):
        write(Path(git.repo_path), "app.py", "def main():\n    return 'changed'\n")
        write(Path(git.repo_path), "brand_new.txt", "hi\n")
        (Path(git.repo_path) / "pyproject.toml").unlink()
        by_path = {e["path"]: e for e in await git.status()}
        assert by_path["app.py"]["status"].strip() == "M"
        assert by_path["brand_new.txt"]["status"] == "??"
        assert by_path["pyproject.toml"]["status"].strip() == "D"
        assert await git.is_dirty() is True

    @pytest.mark.asyncio
    async def test_unmerged_flag_is_set_for_a_real_conflict(self, git):
        root = Path(git.repo_path)
        write(root, "shared.txt", "base\n")
        await git.commit_all(root, "base")
        run(root, "checkout", "-b", "other")
        write(root, "shared.txt", "theirs\n")
        await git.commit_all(root, "theirs")
        run(root, "checkout", "main")
        write(root, "shared.txt", "ours\n")
        await git.commit_all(root, "ours")
        run(root, "merge", "other", check=False)
        entries = {e["path"]: e for e in await git.status()}
        assert entries["shared.txt"]["unmerged"] is True

    @pytest.mark.asyncio
    async def test_staged_addition_is_reported(self, git):
        root = Path(git.repo_path)
        write(root, "staged.txt", "x\n")
        run(root, "add", "staged.txt")
        by_path = {e["path"]: e for e in await git.status()}
        assert by_path["staged.txt"]["status"] == "A "

    @pytest.mark.asyncio
    async def test_first_entry_keeps_its_leading_space_status(self, git):
        """
        A worktree-modified entry starts with " M", and that space is part of
        the porcelain format. Trimming it would turn "app.py" into "pp.py".
        """
        root = Path(git.repo_path)
        write(root, "app.py", "def main():\n    return 'changed'\n")
        entries = await git.status()
        assert entries[0]["path"] == "app.py"
        assert entries[0]["status"] == " M"

    @pytest.mark.asyncio
    async def test_every_reported_path_actually_exists_or_was_deleted(self, git):
        root = Path(git.repo_path)
        write(root, "app.py", "changed\n")
        write(root, "new.txt", "new\n")
        (root / "pyproject.toml").unlink()
        for entry in await git.status():
            if entry["status"].strip() == "D":
                continue
            assert (root / entry["path"]).exists(), entry["path"]


class TestCommits:
    @pytest.mark.asyncio
    async def test_commit_all_returns_the_new_sha(self, git):
        root = Path(git.repo_path)
        write(root, "x.txt", "x\n")
        sha = await git.commit_all(root, "add x")
        assert sha and len(sha) == 40
        assert (await git.commit_info()).message == "add x"

    @pytest.mark.asyncio
    async def test_commit_all_returns_none_when_nothing_changed(self, git):
        assert await git.commit_all(Path(git.repo_path), "no-op") is None

    @pytest.mark.asyncio
    async def test_commit_info_lists_files_and_author(self, git):
        root = Path(git.repo_path)
        write(root, "a.txt", "a\n")
        write(root, "b.txt", "b\n")
        await git.commit_all(root, "two files")
        info: CommitInfo = await git.commit_info()
        assert sorted(info.files) == ["a.txt", "b.txt"]
        assert info.author

    @pytest.mark.asyncio
    async def test_commit_info_of_a_bad_ref_is_empty_not_an_exception(self, git):
        info = await git.commit_info("no-such-ref")
        assert info.sha == ""

    @pytest.mark.asyncio
    async def test_file_at_returns_content_or_none(self, git):
        root = Path(git.repo_path)
        write(root, "a.txt", "alpha\n")
        await git.commit_all(root, "add a")
        assert await git.file_at("HEAD", "a.txt") == "alpha"
        assert await git.file_at("HEAD", "missing.txt") is None

    @pytest.mark.asyncio
    async def test_show_file_does_not_raise(self, git):
        assert await git.show_file("HEAD", "nope.txt") == ""


class TestDiffs:
    @pytest.mark.asyncio
    async def test_diff_between_two_commits(self, git):
        root = Path(git.repo_path)
        base = await git.head_commit()
        write(root, "app.py", "def main():\n    return 'one'\n")
        await git.commit_all(root, "change")
        stat = await git.diff(base)
        assert stat.files_changed == ["app.py"]
        assert stat.insertions >= 1
        assert "return 'one'" in stat.patch
        assert stat.is_empty is False

    @pytest.mark.asyncio
    async def test_diff_against_itself_is_empty(self, git):
        assert (await git.diff("HEAD", "HEAD")).is_empty is True

    @pytest.mark.asyncio
    async def test_diff_working_tree_reports_uncommitted_changes(self, git):
        root = Path(git.repo_path)
        write(root, "app.py", "def main():\n    return 'dirty'\n")
        stat = await git.diff_working_tree()
        assert stat.files_changed == ["app.py"]
        assert stat.patch.strip()
        assert stat.insertions >= 1

    @pytest.mark.asyncio
    async def test_diff_working_tree_is_empty_on_a_clean_tree(self, git):
        assert (await git.diff_working_tree()).is_empty is True

    @pytest.mark.asyncio
    async def test_diff_working_tree_counts_deletions(self, git):
        root = Path(git.repo_path)
        (root / "pyproject.toml").write_text("a\nb\nc\nd\n", encoding="utf-8")
        run(root, "add", "-A")
        run(root, "commit", "-m", "longer")
        (root / "pyproject.toml").write_text("a\n", encoding="utf-8")
        stat = await git.diff_working_tree()
        assert stat.deletions == 3

    @pytest.mark.asyncio
    async def test_stat_only_omits_the_patch(self, git):
        root = Path(git.repo_path)
        base = await git.head_commit()
        write(root, "app.py", "x = 1\n")
        await git.commit_all(root, "change")
        stat = await git.diff(base, stat_only=True)
        assert stat.files_changed == ["app.py"]
        assert stat.patch == ""

    @pytest.mark.asyncio
    async def test_diff_of_two_unknown_refs_is_empty(self, git):
        assert (await git.diff("nope", "also-nope")).is_empty is True


class TestWorktrees:
    @pytest.mark.asyncio
    async def test_create_worktree_checks_out_the_new_branch(self, git, tmp_path):
        target = git.create_worktree
        path = await target("agent/one", tmp_path / "wt1")
        assert path.is_dir()
        assert (path / "app.py").exists()
        assert await git.branch_exists("agent/one") is True

    @pytest.mark.asyncio
    async def test_worktree_content_matches_the_base_point(self, git, tmp_path):
        root = Path(git.repo_path)
        write(root, "app.py", "base version\n")
        await git.commit_all(root, "v1")
        path = await git.create_worktree("agent/v1", tmp_path / "wt")
        assert (path / "app.py").read_text(encoding="utf-8") == "base version\n"

    @pytest.mark.asyncio
    async def test_two_worktrees_are_isolated(self, git, tmp_path):
        root = Path(git.repo_path)
        a = await git.create_worktree("agent/a", tmp_path / "a")
        b = await git.create_worktree("agent/b", tmp_path / "b")
        (a / "only_in_a.txt").write_text("a\n", encoding="utf-8")
        (b / "only_in_b.txt").write_text("b\n", encoding="utf-8")
        assert not (a / "only_in_b.txt").exists()
        assert not (b / "only_in_a.txt").exists()

    @pytest.mark.asyncio
    async def test_reusing_an_existing_branch_checks_it_out(self, git, tmp_path):
        first = await git.create_worktree("agent/shared", tmp_path / "one")
        write(first, "made.txt", "hello\n")
        await git.commit_all(first, "work in agent")
        await git.remove_worktree(first)
        second = await git.create_worktree("agent/shared", tmp_path / "two")
        assert (second / "made.txt").exists()

    @pytest.mark.asyncio
    async def test_remove_worktree_deletes_the_directory(self, git, tmp_path):
        path = await git.create_worktree("agent/gone", tmp_path / "wt")
        assert await git.remove_worktree(path) is True
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_remove_worktree_survives_a_dirty_tree(self, git, tmp_path):
        path = await git.create_worktree("agent/dirty", tmp_path / "wt")
        (path / "junk.txt").write_text("uncommitted\n", encoding="utf-8")
        assert await git.remove_worktree(path) is True

    @pytest.mark.asyncio
    async def test_remove_worktree_cleans_up_a_directory_git_lost(self, git, tmp_path):
        path = await git.create_worktree("agent/ghost", tmp_path / "wt")
        # Simulate git forgetting about it.
        run(git.repo_path, "worktree", "prune")
        (path / "stray.txt").write_text("x\n", encoding="utf-8")
        assert await git.remove_worktree(path) is True

    @pytest.mark.asyncio
    async def test_list_worktrees_includes_the_main_one(self, git, tmp_path):
        await git.create_worktree("agent/listed", tmp_path / "wt")
        paths = {Path(w["path"]).resolve() for w in await git.list_worktrees()}
        assert Path(git.repo_path).resolve() in paths
        assert (tmp_path / "wt").resolve() in paths
        branches = {w.get("branch") for w in await git.list_worktrees()}
        assert "agent/listed" in branches

    @pytest.mark.asyncio
    async def test_worktree_path_for_is_deterministic(self, git, tmp_path):
        assert git.worktree_path_for(tmp_path, "abc") == Path(tmp_path) / "abc"

    @pytest.mark.asyncio
    async def test_delete_branch_after_its_worktree_is_removed(self, git, tmp_path):
        path = await git.create_worktree("agent/temp", tmp_path / "wt")
        assert await git.delete_branch("agent/temp") is False  # still checked out
        await git.remove_worktree(path)
        assert await git.delete_branch("agent/temp") is True
        assert await git.branch_exists("agent/temp") is False

    @pytest.mark.asyncio
    async def test_delete_branch_of_a_missing_branch_is_false(self, git):
        assert await git.delete_branch("never-existed") is False


class TestMerges:
    @pytest.fixture
    async def diverged(self, git):
        """main and feature both change different files, cleanly mergeable."""
        root = Path(git.repo_path)
        run(root, "checkout", "-b", "feature")
        write(root, "feature.py", "FEATURE\n")
        assert await git.commit_all(root, "feature work")
        run(root, "checkout", "main")
        write(root, "main_only.py", "MAIN\n")
        assert await git.commit_all(root, "main work")
        return root

    @pytest.fixture
    async def conflicting(self, git):
        """main and feature both rewrite the same line."""
        root = Path(git.repo_path)
        write(root, "shared.txt", "original\n")
        assert await git.commit_all(root, "shared base")
        run(root, "checkout", "-b", "feature")
        write(root, "shared.txt", "from feature\n")
        assert await git.commit_all(root, "feature edit")
        run(root, "checkout", "main")
        write(root, "shared.txt", "from main\n")
        assert await git.commit_all(root, "main edit")
        return root

    @pytest.mark.asyncio
    async def test_clean_merge_creates_a_two_parent_commit(self, diverged, git):
        outcome = await git.merge("feature", "main")
        assert outcome.success is True
        assert outcome.conflicts == []
        parents = run(diverged, "rev-list", "--parents", "-n", "1", outcome.commit)
        assert len(parents.split()) == 3  # commit + two parents

    @pytest.mark.asyncio
    async def test_clean_merge_brings_both_sides_in(self, diverged, git):
        await git.merge("feature", "main")
        root = Path(git.repo_path)
        assert (root / "feature.py").exists()
        assert (root / "main_only.py").exists()

    @pytest.mark.asyncio
    async def test_merge_defaults_to_the_default_branch(self, diverged, git):
        outcome = await git.merge("feature")
        assert outcome.success is True
        assert await git.current_branch() == "main"

    @pytest.mark.asyncio
    async def test_merge_of_a_missing_branch_fails_cleanly(self, git):
        outcome = await git.merge("no-such-branch", "main")
        assert outcome.success is False
        assert "does not exist" in outcome.message

    @pytest.mark.asyncio
    async def test_merging_twice_reports_already_merged(self, diverged, git):
        await git.merge("feature", "main")
        again = await git.merge("feature", "main")
        assert again.success is True
        assert again.already_merged is True

    @pytest.mark.asyncio
    async def test_conflicting_merge_aborts_and_leaves_a_clean_tree(
        self, conflicting, git
    ):
        outcome = await git.merge("feature", "main")
        assert outcome.success is False
        assert outcome.conflicts == ["shared.txt"]
        assert await git.is_dirty() is False
        assert (Path(git.repo_path) / "shared.txt").read_text(
            encoding="utf-8"
        ) == "from main\n"

    @pytest.mark.asyncio
    async def test_conflict_details_carry_ours_and_theirs(self, conflicting, git):
        outcome = await git.merge("feature", "main")
        detail = outcome.conflict_details[0]
        assert detail["path"] == "shared.txt"
        assert "from main" in detail["ours"]
        assert "from feature" in detail["theirs"]
        assert detail["conflict_count"] == 1

    @pytest.mark.asyncio
    async def test_detect_conflicts_previews_without_touching_the_tree(
        self, conflicting, git
    ):
        conflicts = await git.detect_conflicts("feature", "main")
        assert [c["path"] for c in conflicts] == ["shared.txt"]
        assert conflicts[0]["stages"] >= 2
        assert conflicts[0]["stage1"] != conflicts[0]["stage2"]
        # The preview must not have started a merge.
        assert await git.is_dirty() is False
        assert await git.current_branch() == "main"

    @pytest.mark.asyncio
    async def test_detect_conflicts_is_empty_for_a_clean_merge(self, diverged, git):
        assert await git.detect_conflicts("feature", "main") == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_of_an_unknown_branch_is_empty(self, git):
        assert await git.detect_conflicts("nope", "main") == []

    @pytest.mark.asyncio
    async def test_begin_merge_leaves_the_conflict_in_place(self, conflicting, git):
        outcome = await git.begin_merge("feature", "main")
        assert outcome.success is False
        assert outcome.conflicts == ["shared.txt"]
        content = (Path(git.repo_path) / "shared.txt").read_text(encoding="utf-8")
        assert "<<<<<<<" in content
        await git.abort_merge()
        assert await git.is_dirty() is False

    @pytest.mark.asyncio
    async def test_merge_and_resolve_commits_a_real_merge(self, conflicting, git):
        outcome = await git.merge_and_resolve(
            "feature", {"shared.txt": "combined\n"}, "main"
        )
        assert outcome.success is True
        assert outcome.conflicts == ["shared.txt"]
        assert (Path(git.repo_path) / "shared.txt").read_text(
            encoding="utf-8"
        ) == "combined\n"
        parents = run(
            Path(git.repo_path), "rev-list", "--parents", "-n", "1", outcome.commit
        )
        assert len(parents.split()) == 3  # two parents, not a plain new commit

    @pytest.mark.asyncio
    async def test_merge_and_resolve_aborts_when_a_path_is_missing(
        self, conflicting, git
    ):
        outcome = await git.merge_and_resolve("feature", {}, "main")
        assert outcome.success is False
        assert "no resolution supplied" in outcome.message
        assert "shared.txt" in outcome.message
        assert await git.is_dirty() is False

    @pytest.mark.asyncio
    async def test_merge_and_resolve_passes_through_a_clean_merge(self, diverged, git):
        outcome = await git.merge_and_resolve("feature", {}, "main")
        assert outcome.success is True
        assert outcome.conflicts == []

    @pytest.mark.asyncio
    async def test_a_dirty_target_does_not_block_a_resolved_merge(self, conflicting, git):
        """
        A resolver agent writes its result into the working tree before the
        merge is retried. Git refuses to merge over that, and the subsequent
        rollback would discard the agent's work.
        """
        (Path(git.repo_path) / "shared.txt").write_text("resolved by agent\n", encoding="utf-8")
        outcome = await git.merge_and_resolve(
            "feature", {"shared.txt": "resolved by agent\n"}, "main"
        )
        assert outcome.success is True
        assert (Path(git.repo_path) / "shared.txt").read_text(
            encoding="utf-8"
        ) == "resolved by agent\n"

    @pytest.mark.asyncio
    async def test_resolved_merge_leaves_no_merging_state(self, conflicting, git):
        await git.merge_and_resolve("feature", {"shared.txt": "done\n"}, "main")
        run(Path(git.repo_path), "status")  # must not raise
        assert await git.is_dirty() is False

    @pytest.mark.asyncio
    async def test_split_conflict_markers_handles_multiple_hunks(self):
        content = (
            "a\n<<<<<<< HEAD\nours-1\n=======\ntheirs-1\n>>>>>>> branch\n"
            "b\n<<<<<<< HEAD\nours-2\n=======\ntheirs-2\n>>>>>>> branch\n"
        )
        ours, theirs = GitManager._split_conflict_markers(content)
        assert ours == "ours-1\nours-2"
        assert theirs == "theirs-1\ntheirs-2"

    @pytest.mark.asyncio
    async def test_split_conflict_markers_on_clean_content(self):
        assert GitManager._split_conflict_markers("clean\n") == ("", "")

    @pytest.mark.asyncio
    async def test_parse_merge_tree_groups_stages_per_path(self):
        output = (
            f"{TREE_OID}\n"
            "\n"
            f"100644 {OID_BASE} 1\tone.py\n"
            f"100644 {OID_OURS} 2\tone.py\n"
            f"100644 {OID_THEIRS} 3\tone.py\n"
            "\n"
            "Auto-merging one.py\n"
            "CONFLICT (content): Merge conflict in one.py\n"
        )
        conflicts = GitManager._parse_merge_tree_conflicts(output)
        assert len(conflicts) == 1
        assert conflicts[0]["path"] == "one.py"
        assert conflicts[0]["stages"] == 3
        assert conflicts[0]["stage1"] == OID_BASE
        assert conflicts[0]["stage2"] == OID_OURS
        assert "CONFLICT" in conflicts[0]["message"]

    @pytest.mark.asyncio
    async def test_parse_merge_tree_handles_several_paths(self):
        output = (
            f"{TREE_OID}\n"
            f"100644 {OID_BASE} 1\ta.py\n"
            f"100644 {OID_OURS} 2\ta.py\n"
            f"100644 {OID_THEIRS} 3\ta.py\n"
            f"100644 {OID_BASE} 1\tb.py\n"
            f"100644 {OID_THEIRS} 3\tb.py\n"
        )
        conflicts = GitManager._parse_merge_tree_conflicts(output)
        assert {c["path"] for c in conflicts} == {"a.py", "b.py"}
        by_path = {c["path"]: c for c in conflicts}
        assert by_path["a.py"]["stages"] == 3
        assert by_path["b.py"]["stages"] == 2

    @pytest.mark.asyncio
    async def test_parse_merge_tree_of_a_clean_merge_is_empty(self):
        assert GitManager._parse_merge_tree_conflicts(f"{TREE_OID}\n") == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_records_three_stages(self, conflicting, git):
        conflicts = await git.detect_conflicts("feature", "main")
        assert conflicts[0]["stages"] == 3
        assert conflicts[0]["stage2"] != conflicts[0]["stage3"]


class TestRecovery:
    @pytest.mark.asyncio
    async def test_reset_hard_discards_changes(self, git):
        root = Path(git.repo_path)
        write(root, "app.py", "wrecked\n")
        write(root, "untracked.txt", "junk\n")
        await git.reset_hard()
        assert "def main" in (root / "app.py").read_text(encoding="utf-8")
        assert not (root / "untracked.txt").exists()

    @pytest.mark.asyncio
    async def test_revert_to_restores_a_known_good_commit(self, git):
        root = Path(git.repo_path)
        good = await git.head_commit()
        write(root, "app.py", "broken\n")
        await git.commit_all(root, "bad")
        await git.revert_to(good)
        assert await git.head_commit() == good
        assert "def main" in (root / "app.py").read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_snapshot_ref_is_the_current_head(self, git):
        assert await git.snapshot_ref() == await git.head_commit()

    @pytest.mark.asyncio
    async def test_fetch_of_a_missing_remote_reports_false(self, git):
        assert await git.fetch("origin") is False


class TestHelpers:
    def test_slugify(self):
        assert _slugify("Add User Authentication!") == "add-user-authentication"
        assert _slugify("  ") == "task"
        assert _slugify("a" * 80) == "a" * 40

    def test_is_git_installed(self):
        assert is_git_installed() is True

    def test_repo_size_hint(self, tmp_path, seed_repo):
        assert repo_size_hint(seed_repo) == "present"
        assert repo_size_hint(tmp_path / "gone") == "missing"
        empty = tmp_path / "bare"
        empty.mkdir()
        assert repo_size_hint(empty) == "empty"

    def test_merge_outcome_defaults(self):
        outcome = MergeOutcome(success=False)
        assert outcome.conflicts == []
        assert outcome.already_merged is False

    def test_diff_stat_defaults(self):
        assert DiffStat().is_empty is True


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_two_worktrees_commit_without_touching_each_other(self, git, tmp_path):
        """The whole point of worktrees: parallel agents do not clobber."""
        root = Path(git.repo_path)
        write(root, "shared_seed.txt", "seed\n")
        await git.commit_all(root, "seed")

        async def work(name: str, filename: str) -> tuple[Path, str | None]:
            path = await git.create_worktree(f"agent/{name}", tmp_path / name)
            write(path, filename, f"{name}\n")
            sha = await git.commit_all(path, f"{name} work")
            return path, sha

        (pa, sha_a), (pb, sha_b) = await asyncio.gather(
            work("alpha", "alpha.py"), work("beta", "beta.py")
        )
        assert sha_a and sha_b and sha_a != sha_b
        assert (pa / "alpha.py").exists()
        assert not (pa / "beta.py").exists()
        assert (pb / "beta.py").exists()
        assert not (pb / "alpha.py").exists()

    @pytest.mark.asyncio
    async def test_merging_both_branches_keeps_both_files(self, git, tmp_path):
        root = Path(git.repo_path)
        write(root, "seed.txt", "seed\n")
        await git.commit_all(root, "seed")
        for name, filename in (("alpha", "alpha.py"), ("beta", "beta.py")):
            path = await git.create_worktree(f"agent/{name}", tmp_path / name)
            write(path, filename, f"{name}\n")
            await git.commit_all(path, f"{name} work")
            await git.remove_worktree(path)
        first = await git.merge("agent/alpha", "main")
        second = await git.merge("agent/beta", "main")
        assert first.success and second.success
        assert (root / "alpha.py").exists()
        assert (root / "beta.py").exists()

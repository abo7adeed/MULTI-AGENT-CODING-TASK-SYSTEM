"""Git layer: async repository, worktree, commit, diff and merge operations."""

from app.git.manager import (
    CommitInfo,
    DiffStat,
    GitError,
    GitManager,
    MergeOutcome,
    is_git_installed,
    repo_size_hint,
)

__all__ = [
    "CommitInfo",
    "DiffStat",
    "GitError",
    "GitManager",
    "MergeOutcome",
    "is_git_installed",
    "repo_size_hint",
]

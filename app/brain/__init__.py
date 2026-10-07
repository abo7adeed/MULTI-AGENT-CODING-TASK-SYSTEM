"""Brain layer: task understanding, repository analysis, decomposition, context.

Everything here decides *what* to do. Nothing here executes or mutates state --
that separation is what keeps the system deterministic where it must be.
"""

from app.brain.analyzer import TaskAnalyzer, TaskSpecification
from app.brain.context import (
    ContextBudget,
    ContextManager,
    RepositorySnapshot,
    detect_conventions,
    scan_repository,
)
from app.brain.decomposer import BLUEPRINTS, Blueprint, TaskDecomposer
from app.brain.repo_analyzer import RepoAnalysis, RepositoryAnalyzer

__all__ = [
    "BLUEPRINTS",
    "Blueprint",
    "ContextBudget",
    "ContextManager",
    "RepoAnalysis",
    "RepositoryAnalyzer",
    "RepositorySnapshot",
    "TaskAnalyzer",
    "TaskDecomposer",
    "TaskSpecification",
    "detect_conventions",
    "scan_repository",
]

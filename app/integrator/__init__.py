"""Integrator layer: collect changes, merge, resolve conflicts, test, review."""

from app.integrator.conflict_resolver import (
    ConflictRegion,
    ConflictResolver,
    GitMarkers,
    ResolutionReport,
    summarise_diff,
)
from app.integrator.system import (
    ChangeCollector,
    ChangeRecord,
    FinalReviewer,
    Integrator,
    RegressionDetector,
)
from app.integrator.test_runner import TestCommand, TestRunner

__all__ = [
    "ChangeCollector",
    "ChangeRecord",
    "ConflictRegion",
    "ConflictResolver",
    "FinalReviewer",
    "GitMarkers",
    "Integrator",
    "RegressionDetector",
    "ResolutionReport",
    "TestCommand",
    "TestRunner",
    "summarise_diff",
]

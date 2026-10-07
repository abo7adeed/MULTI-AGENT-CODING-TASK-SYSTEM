"""
Test-driven calculator module.
"""

from __future__ import annotations

NumberLike = int | float


def add(a: NumberLike, b: NumberLike) -> NumberLike:
    """Return the sum of two numbers."""
    return a + b


def subtract(a: NumberLike, b: NumberLike) -> NumberLike:
    """Return the difference of two numbers."""
    return a - b

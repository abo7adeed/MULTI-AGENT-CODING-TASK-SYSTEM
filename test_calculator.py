"""
Tests for the calculator module.
"""

from __future__ import annotations

import pytest

from calc import add, subtract


class NumberLike:
    """A minimal wrapper to exercise type constraints in calculator functions."""

    def __init__(self, value: float) -> None:
        self.value = value


def test_add_basic() -> None:
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract_basic() -> None:
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5
    assert subtract(-2, -3) == 1


def test_subtract_immutable_inputs() -> None:
    a = NumberLike(5.0)
    b = NumberLike(3.0)
    with pytest.raises(TypeError):
        subtract(a, b)

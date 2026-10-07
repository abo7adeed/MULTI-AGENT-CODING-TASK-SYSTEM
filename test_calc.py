"""Demo calculator tests for the demo-scenario recording.

This file intentionally keeps the name `test_calc.py` because the demo workflow
and documentation reference that exact filename. It exercises the same behavior
as `test_calculator.py` so the demo remains deterministic, but now it actually
exists and imports from the real `calc.py` module.
"""

from __future__ import annotations

import pytest

from calc import add, subtract


def test_add() -> None:
    assert add(2, 3) == 5


def test_subtract() -> None:
    assert subtract(5, 3) == 2


def test_subtract_rejects_non_numeric_inputs() -> None:
    class NumberLike:
        def __init__(self, value: float) -> None:
            self.value = value

    with pytest.raises(TypeError):
        subtract(NumberLike(5.0), NumberLike(3.0))

#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the DeepProbLog builtin predicates."""

import pytest

from deeplog.systems.deepproblog.engine.builtins import _builtin_between
from deeplog.systems.deepproblog.engine.builtins import _builtin_equal
from deeplog.systems.deepproblog.engine.builtins import _builtin_ge
from deeplog.systems.deepproblog.engine.builtins import _builtin_gt
from deeplog.systems.deepproblog.engine.builtins import _builtin_is
from deeplog.systems.deepproblog.engine.builtins import _builtin_le
from deeplog.systems.deepproblog.engine.builtins import _builtin_lt
from deeplog.systems.deepproblog.engine.builtins import _builtin_not_equal
from deeplog.systems.deepproblog.engine.builtins import _evaluate_expression


def _atom(value: str):
    return (value,)


class TestBetween:
    def test_enumerates_range_for_variable(self):
        results = list(_builtin_between(_atom("0"), _atom("2"), ("X",)))
        assert results == [
            {("X",): ("0",)},
            {("X",): ("1",)},
            {("X",): ("2",)},
        ]

    def test_succeeds_for_value_in_range(self):
        assert list(_builtin_between(_atom("1"), _atom("5"), _atom("3"))) == [{}]

    def test_fails_for_value_outside_range(self):
        assert list(_builtin_between(_atom("1"), _atom("5"), _atom("10"))) == []


class TestComparisons:
    def test_equal(self):
        assert list(_builtin_equal(_atom("a"), _atom("a"))) == [{}]
        assert list(_builtin_equal(_atom("a"), _atom("b"))) == []

    def test_not_equal(self):
        assert list(_builtin_not_equal(_atom("a"), _atom("b"))) == [{}]
        assert list(_builtin_not_equal(_atom("a"), _atom("a"))) == []

    def test_lt(self):
        assert list(_builtin_lt(_atom("1"), _atom("2"))) == [{}]
        assert list(_builtin_lt(_atom("2"), _atom("1"))) == []

    def test_gt(self):
        assert list(_builtin_gt(_atom("2"), _atom("1"))) == [{}]
        assert list(_builtin_gt(_atom("1"), _atom("2"))) == []

    def test_le(self):
        assert list(_builtin_le(_atom("1"), _atom("1"))) == [{}]
        assert list(_builtin_le(_atom("2"), _atom("1"))) == []

    def test_ge(self):
        assert list(_builtin_ge(_atom("1"), _atom("1"))) == [{}]
        assert list(_builtin_ge(_atom("1"), _atom("2"))) == []


class TestEvaluateExpression:
    def test_int_atom(self):
        assert _evaluate_expression(_atom("42")) == 42

    def test_float_atom(self):
        assert _evaluate_expression(_atom("1.5")) == 1.5

    def test_addition(self):
        assert _evaluate_expression(("+", _atom("2"), _atom("3"))) == 5

    def test_subtraction(self):
        assert _evaluate_expression(("-", _atom("5"), _atom("3"))) == 2

    def test_integer_division(self):
        assert _evaluate_expression(("div", _atom("7"), _atom("2"))) == 3

    def test_modulo(self):
        assert _evaluate_expression(("mod", _atom("7"), _atom("3"))) == 1

    def test_nested_expression(self):
        expr = ("+", _atom("2"), ("-", _atom("5"), _atom("1")))
        assert _evaluate_expression(expr) == 6

    def test_unsupported_arity_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            _evaluate_expression(("+", _atom("1"), _atom("2"), _atom("3")))

    def test_unsupported_operator_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            _evaluate_expression(("*", _atom("2"), _atom("3")))


class TestBuiltinIs:
    def test_binds_variable_to_integer_result(self):
        result = list(_builtin_is(("X",), ("+", _atom("2"), _atom("3"))))
        assert result == [{("X",): ("5",)}]

    def test_binds_variable_to_float_result(self):
        result = list(_builtin_is(("Y",), _atom("1.5")))
        assert result == [{("Y",): ("1.5",)}]

    def test_succeeds_when_both_sides_evaluate_equal(self):
        result = list(_builtin_is(_atom("5"), ("+", _atom("2"), _atom("3"))))
        assert result == [{}]

    def test_fails_when_both_sides_evaluate_unequal(self):
        result = list(_builtin_is(_atom("7"), ("+", _atom("2"), _atom("3"))))
        assert result == []

    def test_raises_when_rhs_is_variable(self):
        with pytest.raises(ValueError, match="cannot be a variable"):
            list(_builtin_is(_atom("1"), ("X",)))

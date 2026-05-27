#  Copyright (c) 2024-2026. KU Leuven
"""Builtin predicates exposed to the DeepProbLog-style engine."""

from collections.abc import Iterable

from ....symbol import Symbol
from ....symbol import is_variable


def _builtin_between(
    low: Symbol, high: Symbol, value: Symbol
) -> Iterable[dict[Symbol, Symbol]]:
    low_int = int(low[0])
    high_int = int(high[0])
    if is_variable(value):
        for v in range(low_int, high_int + 1):
            yield {value: (str(v),)}
    else:
        value_int = int(value[0])
        if low_int <= value_int <= high_int:
            yield {}


def _builtin_equal(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs == rhs:
        yield {}


def _builtin_not_equal(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs != rhs:
        yield {}


def _builtin_lt(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs < rhs:
        yield {}


def _builtin_gt(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs > rhs:
        yield {}


def _builtin_le(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs <= rhs:
        yield {}


def _builtin_ge(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs >= rhs:
        yield {}


def _evaluate_expression(expression: Symbol):
    if len(expression) == 1:
        try:
            return int(expression[0])
        except ValueError:
            return float(expression[0])
    if len(expression) != 3:
        raise ValueError(f"Expression {expression} is not supported.")
    if expression[0] == "+":
        return _evaluate_expression(expression[1]) + _evaluate_expression(expression[2])
    elif expression[0] == "-":
        return _evaluate_expression(expression[1]) - _evaluate_expression(expression[2])
    elif expression[0] == "div":
        return _evaluate_expression(expression[1]) // _evaluate_expression(
            expression[2]
        )
    elif expression[0] == "mod":
        return _evaluate_expression(expression[1]) % _evaluate_expression(expression[2])
    raise ValueError(f"Expression {expression} is not supported.")


def _builtin_is(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if is_variable(rhs):
        raise ValueError("Second argument for is/2 cannot be a variable")
    if is_variable(lhs):
        yield {lhs: (str(_evaluate_expression(rhs)),)}
    else:
        if _evaluate_expression(lhs) == _evaluate_expression(rhs):
            yield {}


all_builtins = {
    ("between", 3): _builtin_between,
    ("==", 2): _builtin_equal,
    ("\\==", 2): _builtin_not_equal,
    ("<", 2): _builtin_lt,
    (">", 2): _builtin_gt,
    (">=", 2): _builtin_ge,
    ("<=", 2): _builtin_le,
    ("is", 2): _builtin_is,
}

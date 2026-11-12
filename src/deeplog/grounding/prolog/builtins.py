#  Copyright (c) 2024-2026. KU Leuven
"""Builtin predicates exposed to the Prolog grounder.

Arithmetic is SWI-Prolog's under its default flags, so
:class:`~deeplog.grounding.prolog.SimpleGrounder` proves what
:class:`~deeplog.grounding.prolog.JanusGrounder` does. A numeral is read as an
integer or a float, an expression applies the arithmetic operators the Prolog
parser reads, and a number is written back as SWI-Prolog writes it. ``/`` of
two integers is an integer where it divides exactly, ``4 / 2`` is ``2`` and
``3 / 2`` is ``1.5``; ``//`` and ``rem`` round toward zero, ``div`` and ``mod``
toward negative infinity; and an integer compares with a float as a float.
"""

import math
import operator
import re
from collections.abc import Callable
from collections.abc import Iterable
from decimal import Decimal

from deeplog.symbol import Symbol
from deeplog.symbol import calculate_mgu
from deeplog.symbol import is_variable
from deeplog.symbol import symbol_to_pretty_string


type Number = int | float

_INTEGER = re.compile(r"[-+]?\d+")
_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


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


def _builtin_unify(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    substitution = calculate_mgu(lhs, rhs)
    if substitution is not None:
        yield substitution


def _builtin_not_unifiable(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if calculate_mgu(lhs, rhs) is None:
        yield {}


def _builtin_identical(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs == rhs:
        yield {}


def _builtin_not_identical(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if lhs != rhs:
        yield {}


def _arithmetic_comparison(
    compare: Callable[[Number, Number], bool],
) -> Callable[[Symbol, Symbol], Iterable[dict[Symbol, Symbol]]]:
    """The builtin comparing the numbers its two arguments evaluate to with ``compare``."""

    def builtin(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
        left, right = _evaluate_expression(lhs), _evaluate_expression(rhs)
        if isinstance(left, float) or isinstance(right, float):
            left, right = float(left), float(right)
        if compare(left, right):
            yield {}

    return builtin


def _builtin_is(lhs: Symbol, rhs: Symbol) -> Iterable[dict[Symbol, Symbol]]:
    if is_variable(rhs):
        raise ValueError("Second argument for is/2 cannot be a variable")
    value = _evaluate_expression(rhs)
    if is_variable(lhs):
        yield {lhs: _numeral(value)}
    else:
        number = _number(lhs)
        if type(number) is type(value) and number == value:
            yield {}


def _evaluate_expression(expression: Symbol) -> Number:
    """The number ``expression`` evaluates to.

    Raises:
        ValueError: If ``expression`` holds a variable, or a term that is
            neither a numeral nor an arithmetic operator the Prolog parser reads,
            or it raises a negative number to a fraction.
        TypeError: If an operator that takes integers is given a float.
        ZeroDivisionError: If it divides by zero.
        OverflowError: If a float it computes is not finite.
    """
    if len(expression) == 1:
        number = _number(expression)
        if number is None:
            raise ValueError(f"{symbol_to_pretty_string(expression)} is not a number.")
        return number
    function = _FUNCTIONS.get((expression[0], len(expression) - 1))
    if function is None:
        raise ValueError(
            f"Expression {symbol_to_pretty_string(expression)} is not supported."
        )
    value = function(*(_evaluate_expression(argument) for argument in expression[1:]))
    if isinstance(value, float) and not math.isfinite(value):
        raise OverflowError(
            f"{symbol_to_pretty_string(expression)} is not a finite number."
        )
    return value


def _number(term: Symbol) -> Number | None:
    """The number the numeral ``term`` writes, or ``None`` where it is not a numeral."""
    if len(term) == 1:
        if _INTEGER.fullmatch(term[0]):
            return int(term[0])
        if _FLOAT.fullmatch(term[0]):
            return float(term[0])
    return None


def _numeral(number: Number) -> Symbol:
    """``number`` written as SWI-Prolog writes it.

    An integer is written by its digits. A float is written by the fewest digits
    that read back as it, with a fractional part, ``2.0``, and with an exponent
    where it is below 0.0001 or its digits end more than three zeros before the
    point: ``1.0e-05``, ``1.0e+04``.
    """
    if isinstance(number, int):
        return (str(number),)
    decimal = Decimal(repr(number)).normalize()
    digits = "".join(map(str, decimal.as_tuple().digits))
    point = decimal.adjusted() + 1
    if point < -3 or point - len(digits) > 3:
        written = f"{digits[0]}.{digits[1:] or '0'}e{point - 1:+03d}"
    elif point <= 0:
        written = f"0.{'0' * -point}{digits}"
    elif point < len(digits):
        written = f"{digits[:point]}.{digits[point:]}"
    else:
        written = f"{digits}{'0' * (point - len(digits))}.0"
    return (f"{'-' if decimal.is_signed() else ''}{written}",)


def _integers(
    function: Callable[[int, int], Number],
) -> Callable[[Number, Number], Number]:
    """``function``, refusing a float operand."""

    def on_integers(a: Number, b: Number) -> Number:
        if isinstance(a, float) or isinstance(b, float):
            raise TypeError(f"Expected two integers, found {a} and {b}.")
        return function(a, b)

    return on_integers


def _divide(a: Number, b: Number) -> Number:
    """``a / b``: an integer where both are integers and ``b`` divides ``a``, else the quotient of the two as floats."""
    if isinstance(a, int) and isinstance(b, int) and b != 0 and a % b == 0:
        return a // b
    return float(a) / float(b)


def _truncating_divide(a: int, b: int) -> int:
    """``a / b`` rounded toward zero."""
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def _power(a: Number, b: Number) -> Number:
    """``a ^ b``.

    It is ``1`` where ``b`` is zero or ``a`` the integer ``1``, and ``0`` where
    ``a`` is the integer ``0``. Otherwise it is an integer where both are
    integers and ``b`` is positive or ``a`` is ``-1``, and a float elsewhere.

    Raises:
        ZeroDivisionError: If ``a`` is zero and ``b`` negative.
    """
    if b == 0 or (isinstance(a, int) and a == 1):
        return 1
    if a == 0 and b < 0:
        raise ZeroDivisionError("0 cannot be raised to a negative power.")
    if isinstance(a, int) and a == 0:
        return 0
    if isinstance(a, int) and isinstance(b, int):
        return a**-b if b < 0 and a == -1 else a**b
    return math.pow(a, b)


def _shift(a: int, b: int) -> int:
    """``a`` shifted ``b`` bits to the left, or ``-b`` bits to the right where ``b`` is negative."""
    return a << b if b >= 0 else a >> -b


#: The function of each arithmetic operator the Prolog parser reads, by functor
#: and arity.
_FUNCTIONS: dict[tuple[str, int], Callable[[Number, Number], Number]] = {
    ("+", 2): operator.add,
    ("-", 2): operator.sub,
    ("*", 2): operator.mul,
    ("/", 2): _divide,
    ("//", 2): _integers(_truncating_divide),
    ("rem", 2): _integers(lambda a, b: a - b * _truncating_divide(a, b)),
    ("mod", 2): _integers(operator.mod),
    ("div", 2): _integers(operator.floordiv),
    ("^", 2): _power,
    ("/\\", 2): _integers(operator.and_),
    ("\\/", 2): _integers(operator.or_),
    ("xor", 2): _integers(operator.xor),
    ("<<", 2): _integers(_shift),
    (">>", 2): _integers(lambda a, b: _shift(a, -b)),
}


all_builtins = {
    ("between", 3): _builtin_between,
    ("=", 2): _builtin_unify,
    ("\\=", 2): _builtin_not_unifiable,
    ("==", 2): _builtin_identical,
    ("\\==", 2): _builtin_not_identical,
    ("is", 2): _builtin_is,
    ("=:=", 2): _arithmetic_comparison(operator.eq),
    ("=\\=", 2): _arithmetic_comparison(operator.ne),
    ("<", 2): _arithmetic_comparison(operator.lt),
    (">", 2): _arithmetic_comparison(operator.gt),
    ("=<", 2): _arithmetic_comparison(operator.le),
    (">=", 2): _arithmetic_comparison(operator.ge),
}

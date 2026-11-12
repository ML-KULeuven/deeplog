#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the builtin predicates of the Prolog grounder."""

import pytest

from deeplog.grounding.prolog.builtins import _builtin_between
from deeplog.grounding.prolog.builtins import _builtin_is
from deeplog.grounding.prolog.builtins import _evaluate_expression
from deeplog.grounding.prolog.builtins import _numeral
from deeplog.grounding.prolog.builtins import all_builtins
from deeplog.grounding.prolog.parser import parse_term


def _atom(value: str):
    return (value,)


def _holds(text: str) -> bool:
    """Whether the builtin goal ``text`` succeeds."""
    goal = parse_term(text)
    return bool(list(all_builtins[(goal[0], len(goal) - 1)](*goal[1:])))


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


class TestUnification:
    @pytest.mark.parametrize(
        ("lhs", "rhs", "bindings"),
        [
            ("X", "a", {"X": "a"}),
            ("a", "X", {"X": "a"}),
            ("f(X, b)", "f(a, Y)", {"X": "a", "Y": "b"}),
            ("g(X, h(Y))", "g(h(Z), X)", {"X": "h(Z)", "Y": "Z"}),
            ("a", "a", {}),
        ],
    )
    def test_unify_binds_the_variables_of_either_side(self, lhs, rhs, bindings):
        unify = all_builtins[("=", 2)]
        assert list(unify(parse_term(lhs), parse_term(rhs))) == [
            {parse_term(var): parse_term(value) for var, value in bindings.items()}
        ]

    @pytest.mark.parametrize(
        ("lhs", "rhs"), [("a", "b"), ("f(X)", "g(X)"), ("2", "2.0")]
    )
    def test_terms_that_do_not_unify(self, lhs, rhs):
        assert list(all_builtins[("=", 2)](parse_term(lhs), parse_term(rhs))) == []
        assert _holds(f"{lhs} \\= {rhs}")

    def test_terms_that_unify_are_not_not_unifiable(self):
        assert not _holds("f(X) \\= f(b)")


class TestIdentity:
    def test_identical(self):
        assert _holds("a == a")
        assert _holds("f(X) == f(X)")
        assert not _holds("a == b")
        assert not _holds("2 == 2.0")
        assert not _holds("X == a")

    def test_not_identical(self):
        assert _holds("a \\== b")
        assert _holds("X \\== Y")
        assert not _holds("a \\== a")


class TestArithmeticComparison:
    @pytest.mark.parametrize(
        "text", ["10 > 9", "9 < 10", "2 =< 2.0", "2.0 >= 2", "1 =:= 1.0", "1 =\\= 2"]
    )
    def test_numbers_compare_by_value(self, text):
        assert _holds(text)

    @pytest.mark.parametrize("text", ["9 > 10", "10 < 9", "2 =:= 3", "2 =\\= 2.0"])
    def test_a_false_comparison_fails(self, text):
        assert not _holds(text)

    def test_both_sides_are_evaluated(self):
        assert _holds("2 * 3 =:= 12 / 2")
        assert _holds("7 // 2 < 7 / 2")

    def test_an_integer_compares_with_a_float_as_a_float(self):
        """2 ** 53 + 1 has no float of its own, so it reads as 2 ** 53 beside one."""
        assert _holds("9007199254740993 =:= 9007199254740992.0")

    def test_an_unbound_variable_has_no_value(self):
        with pytest.raises(ValueError, match="X is not a number"):
            _holds("X >= 0")


class TestEvaluateExpression:
    def test_int_atom(self):
        assert _evaluate_expression(_atom("42")) == 42

    def test_float_atom(self):
        assert _evaluate_expression(_atom("1.5")) == 1.5

    def test_addition(self):
        assert _evaluate_expression(("+", _atom("2"), _atom("3"))) == 5

    def test_subtraction(self):
        assert _evaluate_expression(("-", _atom("5"), _atom("3"))) == 2

    def test_multiplication(self):
        assert _evaluate_expression(("*", _atom("2"), _atom("3"))) == 6

    def test_nested_expression(self):
        expr = ("+", _atom("2"), ("-", _atom("5"), _atom("1")))
        assert _evaluate_expression(expr) == 6

    @pytest.mark.parametrize(
        ("text", "value"),
        [("4 / 2", 2), ("3 / 2", 1.5), ("(-3) / 2", -1.5), ("4.0 / 2", 2.0)],
    )
    def test_division_of_integers_is_an_integer_where_it_is_exact(self, text, value):
        result = _evaluate_expression(parse_term(text))
        assert result == value
        assert type(result) is type(value)

    @pytest.mark.parametrize(
        ("operator", "value"), [("//", -3), ("rem", -1), ("div", -4), ("mod", 1)]
    )
    def test_integer_division_rounds_as_prolog_does(self, operator, value):
        """``//`` and ``rem`` round toward zero, ``div`` and ``mod`` downward."""
        assert _evaluate_expression(parse_term(f"(-7) {operator} 2")) == value

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("2 ^ 10", 1024),
            ("2 ^ (-1)", 0.5),
            ("(-1) ^ (-3)", -1),
            ("2.0 ^ 3", 8.0),
            ("2.5 ^ 0", 1),
            ("1 ^ 0.5", 1),
            ("0 ^ 0.5", 0),
        ],
    )
    def test_power(self, text, value):
        result = _evaluate_expression(parse_term(text))
        assert result == value
        assert type(result) is type(value)

    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("12 /\\ 10", 8),
            ("12 \\/ 10", 14),
            ("12 xor 10", 6),
            ("1 << 3", 8),
            ("(-16) >> 2", -4),
            ("1 << (-1)", 0),
        ],
    )
    def test_bitwise_operators(self, text, value):
        assert _evaluate_expression(parse_term(text)) == value

    @pytest.mark.parametrize("operator", ["//", "rem", "mod", "div", "/\\", "<<"])
    def test_an_integer_operator_refuses_a_float(self, operator):
        with pytest.raises(TypeError, match="two integers"):
            _evaluate_expression(parse_term(f"7.0 {operator} 2"))

    @pytest.mark.parametrize(
        "text", ["1 / 0", "1.0 / 0", "1 // 0", "1 mod 0", "0 ^ (-1)", "0.0 ^ (-1)"]
    )
    def test_division_by_zero_raises(self, text):
        with pytest.raises(ZeroDivisionError):
            _evaluate_expression(parse_term(text))

    def test_a_float_out_of_range_raises(self):
        with pytest.raises(OverflowError, match="not a finite number"):
            _evaluate_expression(parse_term("1.0e308 * 10"))

    def test_a_variable_is_not_a_number(self):
        with pytest.raises(ValueError, match="X is not a number"):
            _evaluate_expression(parse_term("X + 1"))

    def test_unsupported_arity_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            _evaluate_expression(("+", _atom("1"), _atom("2"), _atom("3")))

    def test_unsupported_operator_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            _evaluate_expression(("**", _atom("2"), _atom("3")))


@pytest.mark.parametrize(
    ("number", "written"),
    [
        (5, "5"),
        (-2, "-2"),
        (2**100, "1267650600228229401496703205376"),
        (2.0, "2.0"),
        (-1.5, "-1.5"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (0.1 + 0.2, "0.30000000000000004"),
        (1 / 3, "0.3333333333333333"),
        (0.0001, "0.0001"),
        (0.00001, "1.0e-05"),
        (1000.0, "1000.0"),
        (10000.0, "1.0e+04"),
        (1234000.0, "1234000.0"),
        (12340000.0, "1.234e+07"),
        (2.0**100, "1.2676506002282294e+30"),
        (5e-324, "5.0e-324"),
    ],
)
def test_a_number_is_written_as_swi_prolog_writes_it(number, written):
    assert _numeral(number) == (written,)


class TestBuiltinIs:
    def test_binds_variable_to_integer_result(self):
        result = list(_builtin_is(("X",), ("+", _atom("2"), _atom("3"))))
        assert result == [{("X",): ("5",)}]

    def test_binds_variable_to_float_result(self):
        result = list(_builtin_is(("Y",), _atom("1.5")))
        assert result == [{("Y",): ("1.5",)}]

    def test_binds_variable_to_a_quotient(self):
        assert list(_builtin_is(("Y",), parse_term("3 / 2"))) == [{("Y",): ("1.5",)}]
        assert list(_builtin_is(("Y",), parse_term("4 / 2"))) == [{("Y",): ("2",)}]

    def test_succeeds_when_both_sides_evaluate_equal(self):
        result = list(_builtin_is(_atom("5"), ("+", _atom("2"), _atom("3"))))
        assert result == [{}]

    def test_fails_when_both_sides_evaluate_unequal(self):
        result = list(_builtin_is(_atom("7"), ("+", _atom("2"), _atom("3"))))
        assert result == []

    def test_the_left_side_must_be_the_number_itself(self):
        """Prolog unifies a number only with a number of its type and value."""
        assert list(_builtin_is(_atom("5.0"), parse_term("2 + 3"))) == []
        assert list(_builtin_is(parse_term("X + 1"), _atom("3"))) == []

    def test_raises_when_rhs_is_variable(self):
        with pytest.raises(ValueError, match="cannot be a variable"):
            list(_builtin_is(_atom("1"), ("X",)))

#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the plain-Prolog clause splitting and parsing."""

import pytest

from deeplog import parse_symbol
from deeplog.grounding.prolog import split_list
from deeplog.grounding.prolog import str_to_rules
from deeplog.grounding.prolog.parser import iter_clauses
from deeplog.grounding.prolog.parser import parse_term


def test_one_clause_per_line():
    rules = tuple(str_to_rules("a.\nb.\nq :- a, b."))
    assert len(rules) == 3


def test_multiple_clauses_share_a_line():
    rules = tuple(str_to_rules("a. b. q :- a, b."))
    assert len(rules) == 3
    assert rules == tuple(str_to_rules("a.\nb.\nq :- a, b."))


def test_period_inside_arguments_does_not_split():
    rules = tuple(str_to_rules("p(0.5). q(1.5)."))
    assert len(rules) == 2
    assert rules[0][1] == ("p", ("0.5",))


def test_comments_and_blank_lines_are_skipped():
    rules = tuple(str_to_rules("# a comment\n\na. b.\n"))
    assert len(rules) == 2


def test_missing_terminator_is_an_error():
    with pytest.raises(AssertionError):
        tuple(str_to_rules("a. b"))


def test_iter_clauses_yields_terminated_strings():
    assert list(iter_clauses("a. q :- a.")) == ["a.", "q :- a."]


def test_a_label_operator_inside_an_argument_parses():
    """``::`` is an ordinary binary term, wherever it appears."""
    (rule,) = str_to_rules("p(a::b).")
    assert rule[1] == ("p", ("::", ("a",), ("b",)))


def test_every_anonymous_variable_is_a_variable_of_its_own():
    """Each ``_`` gets a name of its own, and no name the clause already uses."""
    (rule,) = str_to_rules("q(X) :- r(X, _), r(_1, _).")
    assert rule[1] == ("q", ("X",))
    assert rule[2] == (",", ("r", ("X",), ("_2",)), ("r", ("_1",), ("_3",)))


@pytest.mark.parametrize(
    ("symbol_str", "elements", "tail"),
    [
        ("[a,b]", ["a", "b"], "[]"),
        ("[a,[b,c],d|T]", ["a", "[b,c]", "d"], "T"),
        ("[]", [], "[]"),
        ("f(a)", [], "f(a)"),
    ],
    ids=["list", "nested list with a tail", "empty list", "not a list"],
)
def test_split_list(symbol_str, elements, tail):
    """A list splits into its elements and the tail they end in."""
    assert split_list(parse_symbol(symbol_str)) == (
        [parse_symbol(element) for element in elements],
        parse_symbol(tail),
    )


def test_a_term_is_read_around_its_operator():
    assert parse_term("X is Y") == ("is", ("X",), ("Y",))
    assert parse_term("a , b , c") == (",", ("a",), (",", ("b",), ("c",)))


@pytest.mark.parametrize(
    ["text", "term"],
    [
        ("f((a,b))", ("f", (",", ("a",), ("b",)))),
        ("f((a;b))", ("f", (";", ("a",), ("b",)))),
        ("f(a is b)", ("f", ("is", ("a",), ("b",)))),
        ("(a;b),c", (",", (";", ("a",), ("b",)), ("c",))),
        ("(a,b),c", (",", (",", ("a",), ("b",)), ("c",))),
        ("a,b;c", (";", (",", ("a",), ("b",)), ("c",))),
        ("a:-(b;c),d", (":-", ("a",), (",", (";", ("b",), ("c",)), ("d",)))),
        ("(a is b) is c", ("is", ("is", ("a",), ("b",)), ("c",))),
    ],
)
def test_brackets_group_as_in_prolog(text, term):
    """The text is SWI-Prolog's ``writeq`` of the term, up to spaces around ``is``."""
    assert parse_term(text) == term


@pytest.mark.parametrize(
    ["text", "term"],
    [
        ("N1 + N2", "+(N1,N2)"),
        ("Z is N1 + N2", "is(Z,+(N1,N2))"),
        ("a + b * c", "+(a,*(b,c))"),
        ("a * b + c", "+(*(a,b),c)"),
        ("a - b - c", "-(-(a,b),c)"),
        ("(a + b) * c", "*(+(a,b),c)"),
        ("a ^ b ^ c", "^(a,^(b,c))"),
        ("(N1 + N2) mod 10", "mod(+(N1,N2),10)"),
        ("model + 1", "+(model,1)"),
        ("1.5e-3 + X", "+(1.5e-3,X)"),
        ("digit(X,N1) + digit(Y,N2)", "+(digit(X,N1),digit(Y,N2))"),
        ("foo/2", "/(foo,2)"),
        ("p(3-1)", "p(-(3,1))"),
    ],
)
def test_infix_arithmetic_is_read_with_prolog_priorities(text, term):
    """Infix arithmetic reads as the terms Prolog reads it as."""
    assert parse_term(text) == parse_term(term)


@pytest.mark.parametrize(
    "text",
    ["1e-5", "-3", "'a-b'", "X - -1", "a =:= b + 1", "2 ** 3", "a+"],
    ids=[
        "exponent",
        "signed number",
        "quoted atom",
        "unary minus",
        "other operator",
        "unread operator",
        "missing operand",
    ],
)
def test_text_that_is_not_only_infix_arithmetic_stays_one_atom(text):
    """Only operands joined by arithmetic operators read as arithmetic."""
    assert parse_term(text) == (text,)

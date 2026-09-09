#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the plain-Prolog clause splitting and parsing."""

import pytest

from deeplog.grounding.prolog import str_to_rules
from deeplog.grounding.prolog.parser import iter_clauses


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

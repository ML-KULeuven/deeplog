#  Copyright (c) 2024-2026. KU Leuven
"""Tests for plain-Prolog program clauses and their construction."""

from deeplog.grounding.prolog import create_fact
from deeplog.grounding.prolog import create_query
from deeplog.grounding.prolog import create_rule
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_fact
from deeplog.grounding.prolog import is_query
from deeplog.grounding.prolog import str_to_rule
from deeplog.grounding.prolog import str_to_rules


def test_str_to_rules():
    expected_rules = {
        create_rule(
            [("addition", ("X",), ("Y",), ("Z",))],
            [
                ("digit", ("X",), ("N1",)),
                ("digit", ("Y",), ("N2",)),
                ("is", ("Z",), ("N1+N2",)),
            ],
        ),
        create_fact(("::", ("nn", ("X",), ("Y",)), ("digit", ("X",), ("Y",)))),
        create_query([("addition", ("a",), ("b",), ("c",))]),
    }
    code = """
    addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), is(Z,N1+N2).
    nn(X,Y) :: digit(X,Y).
    ?- addition(a,b,c).
    """
    assert set(str_to_rules(code)) == expected_rules


def test_str_to_rule_with_lists():
    rule = str_to_rule("p([a,b|T]) :- q([X|Y]), r([c]).")
    expected_head = ("p", ("cons", ("a",), ("cons", ("b",), ("T",))))
    expected_q = ("q", ("cons", ("X",), ("Y",)))
    expected_r = ("r", ("cons", ("c",), ("nil",)))
    assert rule[1] == expected_head
    assert rule[2] == (",", expected_q, expected_r)


def test_a_fact_is_a_rule_with_a_true_body():
    assert is_fact(str_to_rule("a."))
    assert is_fact(create_fact("a"))
    assert not is_fact(str_to_rule("a :- b."))
    assert not is_fact(str_to_rule(":- a."))
    assert not is_fact(str_to_rule("?- a."))


def test_get_constraint_body():
    """A constraint ``:- body.`` is recognized and its body extracted."""
    simple = str_to_rule(":- alarm.")
    assert is_constraint(simple)
    assert get_constraint_body(simple) == ("alarm",)

    conjunctive = str_to_rule(":- a, not(b).")
    assert is_constraint(conjunctive)
    assert get_constraint_body(conjunctive) == (",", ("a",), ("not", ("b",)))


def test_constraint_and_query_are_disjoint():
    """``:-`` constraints and ``?-`` queries never collide (distinguished by functor)."""
    constraint = str_to_rule(":- alarm.")
    query = str_to_rule("?- alarm.")
    assert is_constraint(constraint) and not is_query(constraint)
    assert is_query(query) and not is_constraint(query)

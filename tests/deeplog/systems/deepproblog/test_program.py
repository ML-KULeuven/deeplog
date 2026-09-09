#  Copyright (c) 2024-2026. KU Leuven

from deeplog.grounding import str_to_rule
from deeplog.grounding import str_to_rules
from deeplog.systems.deepproblog import create_labeled_fact
from deeplog.systems.deepproblog import create_query
from deeplog.systems.deepproblog import create_rule
from deeplog.systems.deepproblog import get_constraint_body
from deeplog.systems.deepproblog import is_constraint
from deeplog.systems.deepproblog import is_query
from deeplog.systems.deepproblog.parser import get_fact_atom
from deeplog.systems.deepproblog.parser import get_fact_label
from deeplog.systems.deepproblog.transformation import remove_labeled_rules


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
        create_labeled_fact(("digit", ("X",), ("Y",)), ("nn", ("X",), ("Y",))),
        create_query([("addition", ("a",), ("b",), ("c",))]),
    }
    code = """
    addition(X,Y,Z) :- digit(X,N1), digit(Y,N2), is(Z,N1+N2).
    nn(X,Y) :: digit(X,Y).
    ?- addition(a,b,c).
    """
    assert set(str_to_rules(code)) == expected_rules


def test_remove_labeled_rules():
    program = (str_to_rule("l::a(X):-x(X)."),)
    expected_program = tuple(
        str_to_rules("""
        l::aux0(X).
        a(X) :- x(X), aux0(X).
    """)
    )

    assert tuple(remove_labeled_rules(program)) == expected_program


def test_str_to_rule_with_lists():
    rule = str_to_rule("p([a,b|T]) :- q([X|Y]), r([c]).")
    expected_head = ("p", ("cons", ("a",), ("cons", ("b",), ("T",))))
    expected_q = ("q", ("cons", ("X",), ("Y",)))
    expected_r = ("r", ("cons", ("c",), ("nil",)))
    assert rule[1] == expected_head
    assert rule[2] == (",", expected_q, expected_r)


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


def test_a_label_survives_the_grounders_parser():
    """DeepProbLog reads its label off the term the plain parser produces.

    There is one parser: ``::`` is a binary term to it, and the DeepProbLog
    reading of that term lives in the accessors, not in a second parser.
    """
    fact = str_to_rule("0.8::coin(c1).")
    assert get_fact_label(fact) == ("0.8",)
    assert get_fact_atom(fact) == ("coin", ("c1",))

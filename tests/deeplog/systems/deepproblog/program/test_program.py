#  Copyright (c) 2024-2026. KU Leuven

from deeplog.systems.deepproblog.program import create_fact
from deeplog.systems.deepproblog.program import create_query
from deeplog.systems.deepproblog.program import create_rule
from deeplog.systems.deepproblog.program import remove_labeled_rules
from deeplog.systems.deepproblog.program import str_to_rule
from deeplog.systems.deepproblog.program import str_to_rules


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
        create_fact(("digit", ("X",), ("Y",)), ("nn", ("X",), ("Y",))),
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

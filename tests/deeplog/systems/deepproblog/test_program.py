#  Copyright (c) 2024-2026. KU Leuven

from deeplog.grounding import str_to_rule
from deeplog.grounding import str_to_rules
from deeplog.systems.deepproblog.parser import get_fact_atom
from deeplog.systems.deepproblog.parser import get_fact_label
from deeplog.systems.deepproblog.transformation import remove_labeled_rules


def test_remove_labeled_rules():
    program = (str_to_rule("l::a(X):-x(X)."),)
    expected_program = tuple(
        str_to_rules("""
        l::aux0(X).
        a(X) :- x(X), aux0(X).
    """)
    )

    assert tuple(remove_labeled_rules(program)) == expected_program


def test_a_label_survives_the_grounders_parser():
    """DeepProbLog reads its label off the term the plain parser produces.

    There is one parser: ``::`` is a binary term to it, and the DeepProbLog
    reading of that term lives in the accessors, not in a second parser.
    """
    fact = str_to_rule("0.8::coin(c1).")
    assert get_fact_label(fact) == ("0.8",)
    assert get_fact_atom(fact) == ("coin", ("c1",))

#  Copyright (c) 2024-2026. KU Leuven
"""A formula's text parses back to the formula."""

from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Transformation
from deeplog import parse_formula


def test_formula_to_text_roundtrip_weighted_formula():
    burglary = ("Burglary",)
    earthquake = ("Earthquake",)
    true = ("true",)

    burglary_bool = Atom(("_", ("=", burglary, true), ("boolean",)))
    earthquake_bool = Atom(("_", ("=", earthquake, true), ("boolean",)))
    transformed = Transformation(
        "probability", BinaryOp("or", burglary_bool, earthquake_bool)
    )

    burglary_prob = Atom(("_", ("p", burglary), ("probability",)))
    earthquake_prob = Atom(("_", ("p", earthquake), ("probability",)))
    joint_probability = BinaryOp("times", burglary_prob, earthquake_prob)
    product = BinaryOp("times", transformed, joint_probability)
    formula = Aggregation("sum", (burglary, earthquake), (), product)

    assert parse_formula(str(formula)) == formula


def test_formula_to_text_preserves_right_associativity():
    a = Atom(("_", ("a",), ("probability",)))
    b = Atom(("_", ("b",), ("probability",)))
    c = Atom(("_", ("c",), ("probability",)))
    formula = BinaryOp("times", a, BinaryOp("times", b, c))

    assert "times (" in str(formula)
    assert parse_formula(str(formula)) == formula


def test_formula_to_text_groups_aggregations_as_operands():
    x = ("X",)
    y = ("Y",)
    aggregation = Aggregation("sum", (x,), (), Atom(("_", ("p", x), ("probability",))))
    formula = BinaryOp("or", aggregation, Atom(("_", ("p", y), ("probability",))))

    assert str(formula).startswith("(sum(")
    assert parse_formula(str(formula)) == formula

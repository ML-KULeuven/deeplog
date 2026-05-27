#  Copyright (c) 2024-2026. KU Leuven
from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula import parse_formula


def test_formula_to_text_roundtrip_weighted_formula():
    factory = SymbolicFormulaFactory()
    burglary = ("Burglary",)
    earthquake = ("Earthquake",)
    true = ("true",)

    burglary_bool = factory.create_atom(("_", ("=", burglary, true), ("boolean",)))
    earthquake_bool = factory.create_atom(("_", ("=", earthquake, true), ("boolean",)))
    boolean_disjunction = factory.create_binary_node(
        "or", burglary_bool, earthquake_bool
    )
    transformed = factory.create_transformation("probability", boolean_disjunction)

    burglary_prob = factory.create_atom(("_", ("p", burglary), ("probability",)))
    earthquake_prob = factory.create_atom(("_", ("p", earthquake), ("probability",)))
    joint_probability = factory.create_binary_node(
        "times", burglary_prob, earthquake_prob
    )
    product = factory.create_binary_node("times", transformed, joint_probability)
    formula = factory.create_aggregation("sum", [burglary, earthquake], [], product)

    reparsed = parse_formula(formula, SymbolicFormulaFactory())

    assert reparsed == formula


def test_formula_to_text_preserves_right_associativity():
    factory = SymbolicFormulaFactory()
    a = factory.create_atom(("_", ("a",), ("probability",)))
    b = factory.create_atom(("_", ("b",), ("probability",)))
    c = factory.create_atom(("_", ("c",), ("probability",)))

    nested = factory.create_binary_node("times", b, c)
    formula = factory.create_binary_node("times", a, nested)

    reparsed = parse_formula(formula, SymbolicFormulaFactory())

    assert "times (" in formula
    assert reparsed == formula


def test_formula_to_text_groups_aggregations_as_operands():
    factory = SymbolicFormulaFactory()
    x = ("X",)
    y = ("Y",)
    agg_child = factory.create_atom(("_", ("p", x), ("probability",)))
    aggregation = factory.create_aggregation("sum", [x], [], agg_child)
    other = factory.create_atom(("_", ("p", y), ("probability",)))
    formula = factory.create_binary_node("or", aggregation, other)

    reparsed = parse_formula(formula, SymbolicFormulaFactory())

    assert formula.startswith("(sum(")
    assert reparsed == formula

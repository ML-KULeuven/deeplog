#  Copyright (c) 2024-2026. KU Leuven
import pytest

from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula import parse_dimacs_cnf


def test_parse_dimacs_boolean_structure():
    dimacs = """
c example
p cnf 3 2
1 -3 0
2 3 -1 0
"""

    factory = SymbolicFormulaFactory()
    parsed = parse_dimacs_cnf(dimacs, factory, structure="boolean")

    x1 = factory.create_atom(("_", ("v1",), ("boolean",)))
    x2 = factory.create_atom(("_", ("v2",), ("boolean",)))
    x3 = factory.create_atom(("_", ("v3",), ("boolean",)))

    clause_one = factory.create_binary_node(
        "or", x1, factory.create_unary_node("not", x3)
    )
    clause_two = factory.create_binary_node(
        "or",
        factory.create_binary_node("or", x2, x3),
        factory.create_unary_node("not", x1),
    )

    expected = factory.create_binary_node("and", clause_one, clause_two)
    assert parsed == expected


def test_parse_dimacs_probability_structure():
    dimacs = """
2 0
-1 0
"""

    factory = SymbolicFormulaFactory()
    parsed = parse_dimacs_cnf(dimacs, factory, structure="probability")

    x2 = factory.create_atom(("_", ("v2",), ("probability",)))
    not_x1 = factory.create_unary_node(
        "negate", factory.create_atom(("_", ("v1",), ("probability",)))
    )

    expected = factory.create_binary_node("times", x2, not_x1)
    assert parsed == expected


def test_reject_missing_clause_terminator():
    factory = SymbolicFormulaFactory()
    with pytest.raises(ValueError, match="terminating 0"):
        parse_dimacs_cnf("1 -2", factory)

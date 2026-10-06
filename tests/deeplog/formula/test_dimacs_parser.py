#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Compiler
from deeplog import SymTensor
from deeplog import UnaryOp
from deeplog import parse_dimacs_cnf
from deeplog import reshape


def _test(variable: str) -> Atom:
    return Atom(("_", ("=", (variable,), ("true",)), ("boolean",)))


def test_a_literal_tests_its_variable():
    dimacs = """
c example
p cnf 3 2
1 -3 0
2 3 -1 0
"""

    parsed = parse_dimacs_cnf(dimacs)

    v1, v2, v3 = _test("V1"), _test("V2"), _test("V3")
    clause_one = BinaryOp("or", v1, UnaryOp("not", v3))
    clause_two = BinaryOp("or", BinaryOp("or", v2, v3), UnaryOp("not", v1))
    assert parsed == BinaryOp("and", clause_one, clause_two)


def test_the_expectation_over_its_variables_is_the_probability_it_holds():
    """``V1 or not V2``, with V1 true at 0.2 and V2 at 0.6: 1 - 0.8 * 0.6."""
    cnf = parse_dimacs_cnf("p cnf 2 1\n1 -2 0")
    expectation = Aggregation("expectation", (("V1",), ("V2",)), (), cnf)

    module = reshape(
        Compiler().compile(expectation),
        input=SymTensor(["=(V1,true) _ probability", "=(V2,true) _ probability"]),
    )

    torch.testing.assert_close(
        module(torch.tensor([[0.2, 0.6]])), torch.tensor([[1 - 0.8 * 0.6]])
    )


def test_reject_missing_clause_terminator():
    with pytest.raises(ValueError, match="terminating 0"):
        parse_dimacs_cnf("1 -2")

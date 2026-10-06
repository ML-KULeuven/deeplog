#  Copyright (c) 2024-2026. KU Leuven
"""Boolean proof formulas, as DeepProbLog's solvers combine them.

``true`` and ``false`` are the identities of conjunction and disjunction, and
are dropped where they are one.
"""

from deeplog import BOOLEAN
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import FormulaNode


TRUE = Atom(("_", BOOLEAN.one, ("boolean",)))
FALSE = Atom(("_", BOOLEAN.zero, ("boolean",)))


def conjoin(lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
    """``lhs`` and ``rhs``, or the one of them that is not :data:`TRUE`."""
    return _combine("and", TRUE, lhs, rhs)


def disjoin(lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
    """``lhs`` or ``rhs``, or the one of them that is not :data:`FALSE`."""
    return _combine("or", FALSE, lhs, rhs)


def _combine(
    operator: str, identity: FormulaNode, lhs: FormulaNode, rhs: FormulaNode
) -> FormulaNode:
    """Apply ``operator``, dropping an operand equal to its ``identity``."""
    if lhs == identity:
        return rhs
    if rhs == identity:
        return lhs
    return BinaryOp(operator, lhs, rhs)

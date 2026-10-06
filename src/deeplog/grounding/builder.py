#  Copyright (c) 2024-2026. KU Leuven
"""The boolean proof formula a grounder builds while it proves a goal.

A grounder emits an ``and`` / ``or`` / ``not`` structure over ground atoms.
``ProofBuilder`` is what it drives to build it, and what a Prolog engine calls
through Janus: it knows only boolean structure and carries no notion of labels,
probabilities or annotated disjunctions.
"""

from __future__ import annotations

from deeplog.algebraic import BOOLEAN
from deeplog.formula.ast import Atom
from deeplog.formula.ast import BinaryOp
from deeplog.formula.ast import FormulaNode
from deeplog.formula.ast import UnaryOp
from deeplog.symbol import Symbol


_TRUE = Atom(("_", BOOLEAN.one, ("boolean",)))
_FALSE = Atom(("_", BOOLEAN.zero, ("boolean",)))


class ProofBuilder:
    """Build a boolean proof formula."""

    def get_true(self) -> FormulaNode:
        """Return the boolean ``true`` constant."""
        return _TRUE

    def get_false(self) -> FormulaNode:
        """Return the boolean ``false`` constant."""
        return _FALSE

    def leaf(self, atom: Symbol) -> FormulaNode:
        """Return the boolean leaf for the ground atom ``atom``."""
        return Atom(("_", atom, ("boolean",)))

    def conjoin(self, lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
        """Combine two proof formulas with boolean conjunction.

        An operand equal to :meth:`get_true` is dropped: it is the identity of
        conjunction.
        """
        return self._combine("and", lhs, rhs, _TRUE)

    def disjoin(self, lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
        """Combine two proof formulas with boolean disjunction.

        An operand equal to :meth:`get_false` is dropped: it is the identity of
        disjunction.
        """
        return self._combine("or", lhs, rhs, _FALSE)

    @staticmethod
    def _combine(
        operator: str, lhs: FormulaNode, rhs: FormulaNode, identity: FormulaNode
    ) -> FormulaNode:
        """Apply ``operator``, dropping an operand equal to its ``identity``."""
        if lhs == identity:
            return rhs
        if rhs == identity:
            return lhs
        return BinaryOp(operator, lhs, rhs)

    def negate(self, operand: FormulaNode) -> FormulaNode:
        """Negate a proof formula with boolean negation."""
        return UnaryOp("not", operand)

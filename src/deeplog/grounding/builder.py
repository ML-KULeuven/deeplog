#  Copyright (c) 2024-2026. KU Leuven
"""A semantics-free boolean proof builder over a :class:`DeepLogFormulaFactory`.

A grounder emits an ``and`` / ``or`` / ``not`` proof structure over ground atoms;
the :class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory` is the
algebra that interprets it (boolean factory -> SAT, circuit factory -> weighted
model count). ``ProofBuilder`` is the thin adapter the grounder drives: it knows
only boolean-structure construction and carries no notion of labels,
probabilities or annotated disjunctions. Extensions that need richer leaves
(e.g. DeepProbLog's probability labels) subclass it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from deeplog.algebraic import BOOLEAN
from deeplog.symbol import Symbol


if TYPE_CHECKING:
    from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory


T = TypeVar("T")


class ProofBuilder[T]:
    """Build a boolean proof structure through a wrapped formula factory."""

    @classmethod
    def wrapping(
        cls, factory: DeepLogFormulaFactory[T] | ProofBuilder[T]
    ) -> ProofBuilder[T]:
        """Return ``factory`` when it already builds proofs, else one over it.

        What a grounder is handed is either the algebra to build through, or a
        builder shared across several calls so that its :attr:`leaves` span all
        of them.
        """
        return factory if isinstance(factory, ProofBuilder) else cls(factory)

    def __init__(self, factory: DeepLogFormulaFactory[T]):
        """Wrap ``factory``; all proof structure is built through it.

        :attr:`leaves` accumulates every ground atom passed to :meth:`leaf` across
        the builder's lifetime -- a semantics-free record of which atoms the
        grounding turned into leaves, that a caller can interpret afterwards.
        """
        self._factory = factory
        self.leaves: set[Symbol] = set()

    def get_true(self) -> T:
        """Return the boolean ``true`` constant leaf."""
        return self._factory.create_atom(("_", BOOLEAN.one, ("boolean",)))

    def get_false(self) -> T:
        """Return the boolean ``false`` constant leaf."""
        return self._factory.create_atom(("_", BOOLEAN.zero, ("boolean",)))

    def leaf(self, atom: Symbol) -> T:
        """Return a free boolean leaf for the ground atom ``atom``."""
        self.leaves.add(atom)
        return self._factory.create_atom(("_", atom, ("boolean",)))

    def conjoin(self, lhs: T, rhs: T) -> T:
        """Combine two proof formulas with boolean conjunction.

        An operand equal to :meth:`get_true` is dropped: it is the identity of
        conjunction.
        """
        return self._combine("and", lhs, rhs, self.get_true())

    def disjoin(self, lhs: T, rhs: T) -> T:
        """Combine two proof formulas with boolean disjunction.

        An operand equal to :meth:`get_false` is dropped: it is the identity of
        disjunction.
        """
        return self._combine("or", lhs, rhs, self.get_false())

    def _combine(self, operator: str, lhs: T, rhs: T, identity: T) -> T:
        """Apply ``operator``, dropping an operand equal to its ``identity``."""
        if lhs == identity:
            return rhs
        if rhs == identity:
            return lhs
        return self._factory.create_binary_node(operator, lhs, rhs)

    def negate(self, operand: T) -> T:
        """Negate a proof formula with boolean negation."""
        return self._factory.create_unary_node("not", operand)

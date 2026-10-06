#  Copyright (c) 2024-2026. KU Leuven
"""A probability-aware proof builder for the k-best prover.

Unlike the semantics-free path (grounder + post-hoc interpretation), the k-best
prover must know probabilities *while* it searches, so it builds labeled leaves
and resolves scalar probabilities in-line. The prover's Prolog engine calls this
builder to make its proof formulas, and it records DeepProbLog's leaf semantics
as it goes -- probability labels and annotated-disjunction tags -- and resolves
scalar probabilities for ranking.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from deeplog import OPEN
from deeplog import PROBABILITY
from deeplog import Atom
from deeplog import Domain
from deeplog import FormulaNode
from deeplog import Symbol
from deeplog import UnaryOp
from deeplog import Variable
from deeplog import VariableAtoms

from ..proofs import FALSE
from ..proofs import TRUE
from ..proofs import conjoin
from ..proofs import disjoin


class ProbabilisticFactory:
    """A proof builder that records probability labels and resolves scalars."""

    def __init__(self, evaluator: Callable[[Symbol], float] | None = None):
        """``evaluator`` resolves non-numeric (neural) labels."""
        self._evaluator = evaluator
        self._labels: dict[Symbol, Symbol] = {}
        self._branches: dict[str, dict[int, Symbol]] = defaultdict(dict)

    @property
    def labels(self) -> dict[Symbol, Symbol]:
        """Return the collected atom-to-label mapping."""
        return self._labels

    @property
    def variables(self) -> VariableAtoms:
        """Where each annotated disjunction the prover reached occurs.

        The prover decides branches *during* search, so unlike the base path
        this is recorded as it goes rather than recognized afterwards. Its
        branches are whatever atoms the disjunction listed, sharing no argument
        position, so the whole atom is the variable's position and the values are
        the atoms themselves. The outcome where none of the branches holds is the
        variable's missing outcome, which no value names.
        """
        recognized: dict[Variable, tuple[Symbol, ...]] = {}
        for cat_id in sorted(self._branches):
            branches = self._branches[cat_id]
            values = [branches[value] for value in sorted(branches)]
            name = ("@variable", values[0])
            recognized[Variable(name, Domain.of(values))] = (OPEN,)
        return recognized

    def get_true(self) -> FormulaNode:
        """Return the boolean ``true`` constant."""
        return TRUE

    def get_false(self) -> FormulaNode:
        """Return the boolean ``false`` constant."""
        return FALSE

    def leaf(self, atom: Symbol) -> FormulaNode:
        """Return the boolean leaf for the ground atom ``atom``."""
        return Atom(("_", atom, ("boolean",)))

    def conjoin(self, lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
        """Combine two proof formulas with boolean conjunction."""
        return conjoin(lhs, rhs)

    def disjoin(self, lhs: FormulaNode, rhs: FormulaNode) -> FormulaNode:
        """Combine two proof formulas with boolean disjunction."""
        return disjoin(lhs, rhs)

    def negate(self, operand: FormulaNode) -> FormulaNode:
        """Negate a proof formula with boolean negation."""
        return UnaryOp("not", operand)

    def get_boolean(self, goal: Symbol, label: Symbol) -> FormulaNode:
        """Create a boolean leaf for ``goal`` and record its probability ``label``."""
        self._labels[goal] = label
        return self.leaf(goal)

    def get_categorical_value(
        self, goal: Symbol, label: Symbol, cat_id: str, value_idx: int
    ) -> FormulaNode:
        """Create a leaf for AD branch ``(cat_id, value_idx)`` reaching ``goal``."""
        self._labels[goal] = label
        self._branches[cat_id][value_idx] = goal
        return self.leaf(goal)

    def get_scalar_probability(self, label: Symbol) -> float:
        """Return a scalar probability for ``label`` (used by k-best's heuristic).

        Numeric-constant labels resolve structurally. Other labels delegate to the
        optional evaluator; if no evaluator was provided, raises ``ValueError``.
        """
        constant = PROBABILITY.get_constant_value(label)
        if constant is not None:
            return float(constant)
        if self._evaluator is None:
            raise ValueError(
                f"Cannot resolve scalar probability for label {label}: not a "
                f"numeric constant and no evaluator was supplied."
            )
        return float(self._evaluator(label))

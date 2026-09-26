#  Copyright (c) 2024-2026. KU Leuven
"""A probability-aware proof builder for the k-best prover.

Unlike the semantics-free path (grounder + post-hoc interpretation), the k-best
prover must know probabilities *while* it searches, so it builds labeled leaves
and resolves scalar probabilities in-line. This factory extends the semantics-free
:class:`~deeplog.grounding.ProofBuilder` with DeepProbLog's leaf semantics --
recording probability labels / annotated-disjunction tags and resolving scalar
probabilities for ranking.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from deeplog import OPEN
from deeplog import Domain
from deeplog import Symbol
from deeplog import Variable
from deeplog import VariableAtoms
from deeplog import get_algebraic_structure
from deeplog.grounding import ProofBuilder


F = TypeVar("F")


class ProbabilisticFactory[F](ProofBuilder[F]):
    """A proof builder that records probability labels and resolves scalars."""

    def __init__(self, factory, evaluator: Callable[[Symbol], float] | None = None):
        """Wrap ``factory``; ``evaluator`` resolves non-numeric (neural) labels."""
        super().__init__(factory)
        self._evaluator = evaluator
        self._labels: dict[Symbol, Symbol] = {}
        self._branches: dict[str, dict[int, Symbol]] = defaultdict(dict)
        self._residuals: dict[str, Symbol] = {}

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
        the atoms themselves. The residual outcome is a value like any other and
        here it has an atom of its own, because the proof formula names it:
        k-best emits a leaf for "the variable took none of the branches" rather
        than the complement.
        """
        recognized: dict[Variable, tuple[Symbol, ...]] = {}
        for cat_id in sorted(set(self._branches) | set(self._residuals)):
            branches = self._branches[cat_id]
            values = [branches[value] for value in sorted(branches)]
            if cat_id in self._residuals:
                values.append(self._residuals[cat_id])
            name = ("@variable", values[0])
            recognized[Variable(name, Domain.of(values))] = (OPEN,)
        return recognized

    def get_boolean(self, goal: Symbol, label: Symbol) -> F:
        """Create a boolean leaf for ``goal`` and record its probability ``label``."""
        self._labels[goal] = label
        return self.leaf(goal)

    def get_categorical_value(
        self, goal: Symbol, label: Symbol, cat_id: str, value_idx: int
    ) -> F:
        """Create a leaf for AD branch ``(cat_id, value_idx)`` reaching ``goal``."""
        self._labels[goal] = label
        self._branches[cat_id][value_idx] = goal
        return self.leaf(goal)

    def get_categorical_none(self, cat_id: str) -> F:
        """Create a leaf for the residual ("no branch chosen") outcome of an AD."""
        none_symbol = ("@cat_none", (cat_id,))
        self._residuals[cat_id] = none_symbol
        return self.leaf(none_symbol)

    def get_scalar_probability(self, label: Symbol) -> float:
        """Return a scalar probability for ``label`` (used by k-best's heuristic).

        Numeric-constant labels resolve structurally. Other labels delegate to the
        optional evaluator; if no evaluator was provided, raises ``ValueError``.
        """
        constant = get_algebraic_structure("probability").get_constant_value(label)
        if constant is not None:
            return float(constant)
        if self._evaluator is None:
            raise ValueError(
                f"Cannot resolve scalar probability for label {label}: not a "
                f"numeric constant and no evaluator was supplied."
            )
        return float(self._evaluator(label))

#  Copyright (c) 2024-2026. KU Leuven
"""Leaf mapping from boolean atoms to their probability labels.

DeepProbLog annotates logical atoms with probability labels (``p :: a`` or a
neural ``nn(...) :: a``). When a boolean proof formula is lowered to the
probability semiring, every boolean leaf must be rewritten to the probability
atom that supplies its value. This module builds that rewrite directly from the
engine's ``labels`` map, so atoms that happen to share arguments stay
unambiguous.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping

from ..symbol import Symbol
from ..symbol import with_structure


def build_leaf_mapping(
    labels: Mapping[Symbol, Symbol],
) -> Callable[[Symbol], Symbol]:
    """Build a boolean-to-probability leaf mapping directly from atom labels.

    ``labels`` maps each labeled boolean atom (e.g. ``("a", ("x1",))``) to its
    probability label atom (e.g. ``("nn1", ("x1",))``). The returned callable
    rewrites a *bare* boolean leaf (the canonical identity a circuit exposes via
    :meth:`~deeplog.circuit.circuit.Circuit.get_leaf_name`) to its matching
    probability leaf. Leaves without an atom label are retagged to the
    probability structure unchanged (they become probability inputs or
    builder-backed leaves).

    Building the mapping straight from ``labels`` keeps it unambiguous when
    distinct atoms share arguments — ``a(x1)`` and ``b(x1)`` labeled by
    ``nn1(x1)`` and ``nn2(x1)`` — which arguments alone cannot resolve.

    A numeric label (e.g. ``0.6 :: fact``) maps to its constant symbol like any
    other, and the target circuit folds it to a constant node.

    Args:
        labels: Maps boolean atoms to their probability label atoms.

    Returns:
        A callable mapping boolean leaf symbols to probability leaf symbols.
    """
    mapping: dict[Symbol, Symbol] = {
        bool_atom: with_structure(prob_atom, "probability")
        for bool_atom, prob_atom in labels.items()
    }

    def leaf_mapping(sym: Symbol) -> Symbol:
        # ``sym`` is a bare boolean leaf; map it to its label or carry it into
        # the probability structure unchanged.
        return mapping.get(sym) or with_structure(sym, "probability")

    return leaf_mapping

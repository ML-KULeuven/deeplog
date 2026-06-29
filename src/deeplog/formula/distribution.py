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

from ..algebraic import get_algebraic_structure
from ..symbol import Symbol
from ..symbol import is_structure_wrapped
from ..symbol import with_structure


def build_leaf_mapping(
    labels: Mapping[Symbol, Symbol],
) -> Callable[[Symbol], Symbol]:
    """Build a boolean-to-probability leaf mapping directly from atom labels.

    ``labels`` maps each labeled boolean atom (e.g. ``("a", ("x1",))``) to its
    probability label atom (e.g. ``("nn1", ("x1",))``). The returned callable
    rewrites a structure-wrapped boolean leaf to its matching probability leaf.
    Leaves without an atom label are retagged to the probability structure
    unchanged (they become probability inputs or builder-backed leaves);
    symbols that are not structure-wrapped pass through untouched.

    Constructing the mapping straight from ``labels`` keeps it unambiguous even
    when distinct atoms share arguments — e.g. ``a(x1)`` and ``b(x1)`` labeled
    by ``nn1(x1)`` and ``nn2(x1)`` — which a by-arguments heuristic cannot
    resolve because both the boolean atoms and their labels collide on
    arguments alone.

    Numeric-constant labels (e.g. ``0.6 :: fact``) are skipped: they would fold
    to a constant node, which the deterministic knowledge-compilation backend
    cannot represent. Such leaves are carried into the probability structure as
    inputs instead; baking numeric facts into the AC is handled separately by
    the categorical/MV-SDD path.

    Args:
        labels: Maps boolean atoms to their probability label atoms.

    Returns:
        A callable mapping boolean leaf symbols to probability leaf symbols.
    """
    probability = get_algebraic_structure("probability")
    mapping: dict[Symbol, Symbol] = {
        with_structure(bool_atom, "boolean"): with_structure(prob_atom, "probability")
        for bool_atom, prob_atom in labels.items()
        if probability.get_constant_value(prob_atom) is None
    }

    def leaf_mapping(sym: Symbol) -> Symbol:
        if sym in mapping:
            return mapping[sym]
        if is_structure_wrapped(sym):
            return with_structure(sym, "probability")
        return sym

    return leaf_mapping

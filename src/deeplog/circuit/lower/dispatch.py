#  Copyright (c) 2024-2026. KU Leuven
"""Backend selection and dispatch for lowering a circuit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...symbol import with_structure
from ..backends import routed_operators
from ..backends import select_backend
from ..split import find_cuts
from ..split import lower_across_cuts
from .generic import lower_generic


if TYPE_CHECKING:
    from ...module import DeepLogModule
    from ...symbol import Symbol
    from ..circuit import Circuit


def to_module(
    circuit: Circuit,
    roots: dict[int, Symbol],
    frontier: Mapping[int, Symbol] | None = None,
) -> DeepLogModule:
    """Lower a Circuit to a DeepLogModule, evaluating it as written.

    Args:
        circuit: The circuit to lower.
        roots: A dictionary mapping node IDs to output names.
        frontier: Nodes to lower as input slots under the given symbols
                 rather than descending into, so nothing beneath them is
                 walked (:mod:`deeplog.circuit.split`).

    Returns:
        A DeepLogModule wrapping the circuit as a torch module.
    """
    # Every root output is a value of the circuit's algebra, so it is labelled
    # here — once, for both backends, which each read their output names only
    # from ``roots.values()``.
    roots = {
        node_id: with_structure(name, circuit.structure.name)
        for node_id, name in roots.items()
    }
    backend = select_backend(circuit)

    # An operator this backend has no node for does not change the backend: the
    # graph is cut there, the operator is applied to the lowered operands, and
    # the circuit above the cut is lowered here like any other.
    cuts = find_cuts(
        circuit, list(roots), routed_operators(circuit.structure, backend), frontier
    )
    if cuts:
        return lower_across_cuts(circuit, roots, cuts, frontier=frontier)

    if backend == "klay":
        from .klay import lower_klay

        return lower_klay(circuit, roots, frontier)
    return lower_generic(circuit, roots, frontier)

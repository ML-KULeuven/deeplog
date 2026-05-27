#  Copyright (c) 2024-2026. KU Leuven
"""PySDD compile backend.

Used for deterministic boolean circuits without categorical (AD) leaves.
Builds an SDD bottom-up via pairwise `&`/`|`/`~`, then walks the canonical
SDD into a Klay circuit and finally a torch module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import klay

from ...module.wrappers import WrappedModule
from ...shape import SymTensor
from .common import klay_to_torch_module
from .common import walk_and_combine


if TYPE_CHECKING:
    from ...module import DeepLogModule
    from ...symbol import Symbol
    from ..circuit import Circuit


def compile_pysdd(
    circuit: Circuit,
    roots: dict[int, Symbol],
    structure_override: str | None = None,
) -> DeepLogModule:
    """Compile a circuit to a PySDD-backed torch module."""
    from pysdd.sdd import SddManager
    from pysdd.sdd import SddNode

    leaf_nodes = circuit.leaf_nodes
    num_vars = len(leaf_nodes)
    sdd_mgr = SddManager(var_count=max(1, num_vars), auto_gc_and_minimize=False)

    node_to_sdd: dict[int, SddNode] = {}

    # Map leaf nodes to SDD literals
    leaf_ids = list(leaf_nodes.values())
    for i, leaf_id in enumerate(leaf_ids):
        node_to_sdd[leaf_id] = sdd_mgr.literal(i + 1)

    # Map constant nodes
    zero_id = circuit.zero_node
    one_id = circuit.one_node
    if zero_id is not None:
        node_to_sdd[zero_id] = sdd_mgr.false()
    if one_id is not None:
        node_to_sdd[one_id] = sdd_mgr.true()

    walk_and_combine(circuit, roots, node_to_sdd)  # pyright: ignore[reportArgumentType]

    root_ids = list(roots.keys())
    klay_circuit = klay.Circuit()
    for root_id in root_ids:
        klay_circuit.add_sdd(node_to_sdd[root_id])

    torch_module = klay_to_torch_module(
        klay_circuit,
        semiring_key=structure_override or circuit.structure,
    )

    # Map leaf symbols to new structure if overridden
    input_symbols = list(leaf_nodes.keys())
    if structure_override:
        input_symbols = [
            (sym[0], sym[1], (structure_override,)) if len(sym) == 3 else sym
            for sym in input_symbols
        ]

    return WrappedModule(
        torch_module,
        SymTensor(input_symbols),
        SymTensor(list(roots.values())),
        name=circuit.name,
        vmap=True,
    )

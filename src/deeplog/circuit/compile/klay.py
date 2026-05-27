#  Copyright (c) 2024-2026. KU Leuven
"""Klay-direct compile backend.

Used for non-deterministic circuits whose operator set is Klay-compatible.
Flattens chains of same-type associative ops into n-ary Klay nodes and drops
identity-constant children to produce a shallow circuit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import klay

from ...module.wrappers import WrappedModule
from ...shape import SymTensor
from .common import klay_to_torch_module


if TYPE_CHECKING:
    from ...module import DeepLogModule
    from ...symbol import Symbol
    from ..circuit import Circuit


def compile_klay(
    circuit: Circuit,
    roots: dict[int, Symbol],
    structure_override: str | None = None,
) -> DeepLogModule:
    """Compile a circuit to a Klay-backed torch module."""
    klay_circuit = klay.Circuit()
    node_to_klay: dict[int, klay.NodePtr] = {}

    # Map leaf nodes to klay literals. Track the signed literal id alongside
    # the NodePtr so we can negate via klay's documented `literal_node(-id)`
    # path; klay has no negate primitive for compound nodes.
    leaf_ids = list(circuit.leaf_nodes.values())
    lit_id_of: dict[int, int] = {}
    for i, leaf_id in enumerate(leaf_ids):
        lit_id_of[leaf_id] = i + 1
        node_to_klay[leaf_id] = klay_circuit.literal_node(i + 1)

    # Map constant nodes
    zero_id = circuit.zero_node
    one_id = circuit.one_node
    if zero_id is not None:
        node_to_klay[zero_id] = klay_circuit.false_node()
    if one_id is not None:
        node_to_klay[one_id] = klay_circuit.true_node()

    root_ids = list(roots.keys())

    _OR_TYPES = frozenset({"or", "plus"})
    _AND_TYPES = frozenset({"and", "times"})
    absorbed, flat_children = circuit.flatten_chains(
        root_ids,
        chain_groups=[
            (_OR_TYPES, frozenset({zero_id} if zero_id is not None else ())),
            (_AND_TYPES, frozenset({one_id} if one_id is not None else ())),
        ],
    )

    for node_id in circuit.iter_topological(root_ids):
        if node_id in node_to_klay or node_id in absorbed:
            continue

        node = circuit.get_node(node_id)

        if node.node_type in _OR_TYPES:
            klay_children = [node_to_klay[c] for c in flat_children[node_id]]
            if not klay_children:
                node_to_klay[node_id] = klay_circuit.false_node()
            elif len(klay_children) == 1:
                node_to_klay[node_id] = klay_children[0]
            else:
                node_to_klay[node_id] = klay_circuit.or_node(klay_children)
        elif node.node_type in _AND_TYPES:
            klay_children = [node_to_klay[c] for c in flat_children[node_id]]
            if not klay_children:
                node_to_klay[node_id] = klay_circuit.true_node()
            elif len(klay_children) == 1:
                node_to_klay[node_id] = klay_children[0]
            else:
                node_to_klay[node_id] = klay_circuit.and_node(klay_children)
        elif node.node_type in ("not", "negate"):
            (child_id,) = node.children
            if child_id not in lit_id_of:
                child_type = circuit.get_node(child_id).node_type
                raise ValueError(
                    f"Klay backend can only negate literals, but '{node.node_type}' "
                    f"is applied to a '{child_type}' node (circuit node {child_id}). "
                    f"Compile via a deterministic backend (PySDD/MV-SDD) which "
                    f"produces NNF, or push negation down to leaves before "
                    f"reaching Klay."
                )
            negated_id = -lit_id_of[child_id]
            lit_id_of[node_id] = negated_id
            node_to_klay[node_id] = klay_circuit.literal_node(negated_id)
        elif node.node_type in ("leaf", "zero", "one"):
            pass
        else:
            raise ValueError(f"Unknown node_type: {node.node_type}")

    for root_id in root_ids:
        klay_circuit.set_root(node_to_klay[root_id])

    torch_module = klay_to_torch_module(
        klay_circuit,
        semiring_key=structure_override or circuit.structure,
        semiring_fallback=(structure_override or "real", False),
    )
    return WrappedModule(
        torch_module,
        SymTensor(list(circuit.leaf_nodes)),
        SymTensor(list(roots.values())),
        name=circuit.name,
        vmap=True,
    )

#  Copyright (c) 2024-2026. KU Leuven
"""Probability distribution helpers for DeepLog formulas.

This module builds factorized probability distributions from labeled atoms,
verifies that probability circuits are fully factorized, and maps boolean
leaves to matching probability leaves by argument overlap.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..circuit.circuit import Circuit
from ..symbol import Symbol
from ..symbol import get_args
from ..symbol import unwrap_structure


if TYPE_CHECKING:
    from .deeplogmodulefactory import DeepLogModuleFactory
    from .deeplogmodulefactory.builder_protocols import InternalNode


def build_probability_distribution(
    labels: Mapping[Symbol, Symbol],
    factory: DeepLogModuleFactory,
) -> InternalNode | None:
    """Build a factorized probability circuit from atom labels.

    Creates a probability atom for each unique labeled atom and combines
    them with ``times`` (fully factorized distribution).

    Args:
        labels: Maps boolean atoms to their probability label atoms.
        factory: The module factory used to create circuit nodes.

    Returns:
        A circuit node representing the factorized probability distribution,
        or ``None`` if no labels are provided.
    """
    from .deeplogmodulefactory.nodes import CompositeCircuitNode

    prob_atoms = set(labels.values())
    if not prob_atoms:
        return None

    prob_nodes: list[InternalNode] = []
    for atom in sorted(prob_atoms, key=str):
        sym = ("_", atom, ("probability",))
        prob_nodes.append(factory.create_atom(sym))

    result = prob_nodes[0]
    for node in prob_nodes[1:]:
        result = factory.create_binary_node("times", result, node)

    # Ensure the result is a circuit node even with a single atom, so that
    # downstream consumers (e.g. _build_expectation) can access .circuit / .node.
    if not isinstance(result, CompositeCircuitNode):
        circuit = factory._get_circuit("probability")
        result = factory._ensure_circuit_node(result, circuit)

    return result


def factorize(circuit: Circuit, root: int) -> list[Symbol]:
    """Verify a probability circuit is fully factorized and return its leaf symbols.

    A fully factorized distribution is a product of independent terms,
    meaning every operator node in the circuit is ``times``.

    Args:
        circuit: The probability circuit to check.
        root: The root node ID.

    Returns:
        A flat list of leaf symbols from the circuit.

    Raises:
        ValueError: If the circuit contains non-product operators.
    """
    for node_id in circuit.iter_topological([root]):
        node = circuit.get_node(node_id)
        if node.node_type in ("leaf", "zero", "one", "constant"):
            continue
        if node.node_type != "times":
            raise ValueError(
                f"Probability formula is not fully factorized: "
                f"found operator '{node.node_type}' (expected only 'times')."
            )
    return list(circuit.leaf_nodes.keys())


def build_leaf_mapping(
    boolean_leaves: list[Symbol],
    probability_leaves: list[Symbol],
) -> Callable[[Symbol], Symbol]:
    """Build a leaf mapping from boolean to probability symbols by argument overlap.

    For each boolean leaf symbol, finds the probability leaf whose arguments
    match. Boolean atoms with no match are mapped to the constant ``("1",)``
    (deterministic, always true).

    Args:
        boolean_leaves: Structure-wrapped leaf symbols from the boolean circuit.
        probability_leaves: Structure-wrapped leaf symbols from the factorized
            probability circuit.

    Returns:
        A callable mapping boolean leaf symbols to probability leaf symbols.

    Raises:
        ValueError: If multiple probability atoms share the same arguments,
            or if a leaf symbol is not structure-wrapped.
    """
    # Index probability leaves by their arguments
    args_to_prob: dict[tuple[Symbol, ...], Symbol] = {}
    for sym in probability_leaves:
        args = get_args(unwrap_structure(sym))
        if not args:
            continue
        if args in args_to_prob:
            raise ValueError(
                f"Ambiguous probability mapping: multiple probability atoms "
                f"share arguments {args}."
            )
        args_to_prob[args] = sym

    # Build the mapping for boolean leaves
    mapping: dict[Symbol, Symbol] = {}
    for sym in boolean_leaves:
        args = get_args(unwrap_structure(sym))
        if args and args in args_to_prob:
            mapping[sym] = args_to_prob[args]
        else:
            mapping[sym] = ("1",)

    def leaf_mapping(sym: Symbol) -> Symbol:
        return mapping.get(sym, sym)

    return leaf_mapping

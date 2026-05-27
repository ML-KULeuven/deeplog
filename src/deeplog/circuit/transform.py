#  Copyright (c) 2024-2026. KU Leuven
"""Transform a circuit from one algebraic structure to another."""

from collections.abc import Callable

from ..algebraic import Algebra
from ..algebraic import AlgebraicStructure
from ..algebraic import Semiring
from ..algebraic import get_algebraic_structure
from ..symbol import Symbol
from .circuit import Circuit


def _build_operator_mapping(
    source: AlgebraicStructure,
    target: AlgebraicStructure,
    explicit: dict[str, str] | None,
) -> dict[str, str]:
    """Build the operator mapping from source to target structure.

    If explicit is provided, use it directly.
    If both are Semiring, auto-infer from roles.
    Otherwise, raise ValueError.
    """
    if explicit is not None:
        for target_op in explicit.values():
            if target_op not in target.operators:
                raise ValueError(
                    f"Target operator '{target_op}' not in target structure "
                    f"'{target.name}'. Available: {target.operators}"
                )
        return explicit

    if not (isinstance(source, Semiring) and isinstance(target, Semiring)):
        raise ValueError(
            f"Cannot auto-map operators between '{source.name}' and "
            f"'{target.name}': both must be Semiring (or Algebra). "
            f"Provide an explicit operator_mapping."
        )

    mapping: dict[str, str] = {
        source.product: target.product,
        source.sum: target.sum,
    }

    if isinstance(source, Algebra):
        if not isinstance(target, Algebra):
            raise ValueError(
                f"Source '{source.name}' has negation ('{source.negation}') "
                f"but target '{target.name}' is not an Algebra. "
                f"Provide an explicit operator_mapping."
            )
        mapping[source.negation] = target.negation

    return mapping


def transform_circuit(
    source: Circuit,
    target_structure: str | AlgebraicStructure,
    roots: list[int],
    *,
    operator_mapping: dict[str, str] | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    deterministic: bool | None = None,
) -> tuple[Circuit, dict[int, int]]:
    """Transform a circuit to a different algebraic structure.

    Creates a new circuit with the target structure by traversing the source
    circuit and rebuilding each node with mapped operators, leaves, and constants.

    Args:
        source: The circuit to transform_circuit.
        target_structure: The target algebraic structure (name or instance).
        roots: Root node IDs defining the subgraph to transform_circuit.
        operator_mapping: Explicit mapping from source operator names to target
            operator names. If None and both structures are Semiring/Algebra,
            mapping is auto-inferred from roles (product→product, sum→sum,
            negation→negation).
        leaf_mapping: Optional callable to remap leaf symbols.
        deterministic: Deterministic flag for the new circuit. If None,
            inherits from the source circuit.

    Returns:
        A tuple of (new_circuit, node_map) where node_map maps source node IDs
        to their corresponding IDs in the new circuit.
    """
    if isinstance(target_structure, str):
        target_struct = get_algebraic_structure(target_structure)
    else:
        target_struct = target_structure

    source_struct = source.algebraic_structure

    op_mapping = _build_operator_mapping(source_struct, target_struct, operator_mapping)

    target_circuit = Circuit(
        target_struct,
        deterministic=deterministic
        if deterministic is not None
        else not target_struct.idempotent,
    )

    target_role_to_symbol = target_struct.named_constants

    # Collect source constant role names for identification during traversal
    source_constant_roles = set(source_struct.named_constants)

    source_constant_values = source.constant_values

    node_map: dict[int, int] = {}

    for source_id in source.iter_topological(roots):
        node = source.get_node(source_id)

        if node.node_type == "leaf":
            name = source.get_leaf_name(source_id)
            if name is None:
                raise ValueError(f"Leaf node {source_id} has no symbol name.")
            mapped_name = leaf_mapping(name) if leaf_mapping else name
            node_map[source_id] = target_circuit.get_leaf_node(mapped_name)

        elif node.node_type in source_constant_roles:
            # Constant role node (e.g., "zero", "one")
            role = node.node_type
            if role in target_role_to_symbol:
                node_map[source_id] = target_circuit.get_leaf_node(
                    target_role_to_symbol[role]
                )
            else:
                raise ValueError(
                    f"Constant role '{role}' from source structure "
                    f"'{source_struct.name}' has no equivalent in target "
                    f"structure '{target_struct.name}'."
                )

        elif node.node_type == "constant":
            # Arbitrary numeric constant
            value = source_constant_values[source_id]
            value_symbol: Symbol = (str(value),)
            node_map[source_id] = target_circuit.get_leaf_node(value_symbol)

        elif node.node_type in op_mapping:
            target_op = op_mapping[node.node_type]
            mapped_children = tuple(node_map[c] for c in node.children)
            apply_op = target_circuit.get_operator(target_op)
            node_map[source_id] = apply_op(*mapped_children)

        else:
            raise ValueError(
                f"Cannot map node type '{node.node_type}' from "
                f"'{source_struct.name}' to '{target_struct.name}'. "
                f"Provide an explicit operator_mapping that includes "
                f"'{node.node_type}'."
            )

    return target_circuit, node_map

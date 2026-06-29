#  Copyright (c) 2024-2026. KU Leuven
"""Factories that lower symbolic formulas to DeepLogModule graphs backed by circuits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial
from typing import cast

import torch

from ...algebraic import AlgebraicStructure
from ...algebraic import HasStructure
from ...circuit import Circuit
from ...circuit import CircuitNode
from ...module import DeepLogModule
from ...module import Sequential
from ...module import SupportsToModule
from ...module import compose_modules
from ...shape import get_only_symbol
from ...symbol import Symbol
from ...symbol import strip_literal_structure
from ..deeplogformulafactory import DeepLogFormulaFactory
from .builder_protocols import AggregationBuilder
from .builder_protocols import AtomBuilder
from .builder_protocols import InternalNode
from .builder_protocols import TransformationBuilder
from .nodes import AtomLeafNode
from .nodes import BuilderLeafNode
from .nodes import CompositeCircuitNode
from .nodes import InputLeafNode
from .registry import default_aggregation_builders
from .registry import default_atom_builders
from .registry import default_circuit_builders
from .registry import default_transformation_builders


class DeepLogModuleFactory(DeepLogFormulaFactory[InternalNode]):
    """Factory that lowers symbolic formulas into DeepLogModule graphs backed by circuits."""

    def __init__(
        self,
        variables: Mapping[Symbol, torch.Tensor] | None = None,
        structures: Mapping[str, AlgebraicStructure] | None = None,
        aggregators: Mapping[str, AggregationBuilder] | None = None,
        atom_builders: Mapping[tuple[str, int, str], AtomBuilder] | None = None,
    ) -> None:
        """Configure the module factory with available domains and default builders."""
        self.variables = defaultdict(lambda: torch.tensor([False, True]))
        self.variables.update(variables or {})

        self._atom_builders = dict(atom_builders or {})
        self._atom_builders.update(default_atom_builders)

        self._circuit_builders = dict(default_circuit_builders)

        self._structures = dict(structures or {})
        for name, structure in self._structures.items():
            self._circuit_builders[name] = partial(Circuit, structure=structure)

        self._aggregation_builders: dict[str, AggregationBuilder] = dict(
            default_aggregation_builders
        )
        self._aggregation_builders.update(aggregators or {})

        self._transformation_builders: dict[tuple[str, str], TransformationBuilder] = {}
        self._transformation_builders.update(default_transformation_builders)

        self._circuits: dict[str, Circuit] = {}

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[InternalNode],
        child: InternalNode,
    ) -> InternalNode:
        """Wrap ``child`` with an aggregation over ``binders`` using ``operation``."""
        domains = [self.variables[var] for var in binders]
        builder = self._aggregation_builders[operation]
        return builder(child, binders, params, domains)

    def create_transformation(
        self, structure: str, child: InternalNode
    ) -> InternalNode:
        """Insert a transformation module to convert ``child`` into ``structure``."""
        child_module = cast(HasStructure, child.to_module())
        key = (child_module.get_structure(), structure)
        builder = self._transformation_builders[key]
        transform_module = Sequential(
            cast(DeepLogModule, child_module),
            builder(cast(DeepLogModule, child_module).get_output_shape()).to_module(),
        )
        transform_module.structure = structure
        return transform_module

    def _get_circuit(self, structure: str) -> Circuit:
        """Get or create the shared circuit for a structure."""
        if structure not in self._circuits:
            self._circuits[structure] = self._circuit_builders[structure]()
        return self._circuits[structure]

    def _ensure_circuit_node(
        self, node: InternalNode, circuit: Circuit
    ) -> CompositeCircuitNode:
        """Wrap ``node`` as a leaf in ``circuit`` if it isn't already part of it."""
        if isinstance(node, CompositeCircuitNode) and node.root.circuit is circuit:
            return node
        children: list[SupportsToModule]
        if isinstance(node, CompositeCircuitNode):
            children = node.children
        elif isinstance(node, InputLeafNode):
            children = []
        else:
            children = [node.to_module()]
        if isinstance(node, AtomLeafNode):
            name = ("_", (node.predicate[0], *node.arguments), (node.predicate[2],))
        elif isinstance(node, CompositeCircuitNode):
            name = node.root.circuit.get_leaf_name(node.root.node)
            if name is None:
                name = get_only_symbol(node.to_module().get_output_shape())
        else:
            name = get_only_symbol(node.to_module().get_output_shape())
        root = CircuitNode(circuit, circuit.get_leaf_node(name))
        return CompositeCircuitNode(root, children)

    def create_binary_node(
        self, operator: str, lhs: InternalNode, rhs: InternalNode
    ) -> InternalNode:
        """Combine two sub-formulas with a binary operator, returning a circuit node."""
        structure = lhs.get_structure()
        circuit = self._get_circuit(structure)
        lhs = self._ensure_circuit_node(lhs, circuit)
        rhs = self._ensure_circuit_node(rhs, circuit)
        new_node = circuit.get_operator(operator)(lhs.root.node, rhs.root.node)
        return CompositeCircuitNode(
            CircuitNode(circuit, new_node), lhs.children + rhs.children
        )

    def create_unary_node(self, operator: str, operand: InternalNode) -> InternalNode:
        """Apply a unary circuit operator to a sub-formula."""
        structure = operand.get_structure()
        circuit = self._get_circuit(structure)
        operand = self._ensure_circuit_node(operand, circuit)
        new_node = circuit.get_operator(operator)(operand.root.node)
        return CompositeCircuitNode(CircuitNode(circuit, new_node), operand.children)

    def create_atom(self, atom: Symbol) -> InternalNode:
        """Create a leaf literal node.

        If a builder is registered for the predicate, the leaf produces a
        sub-module. Otherwise it is an input leaf: its circuit leaf name
        becomes an external input to the composed module.
        """
        if len(atom) != 3 or atom[0] != "_":
            raise ValueError(f"Invalid atom: {atom}")
        predicate, arguments = self._decode_leaf(atom)
        builder = self._atom_builders.get(predicate)
        if builder is None:
            # A bare leaf may turn out to be the whole formula; hand it the
            # circuit builder for its structure so it can compile on its own
            # (identity for a variable leaf, constant for a numeric symbol).
            return InputLeafNode(
                predicate, arguments, self._circuit_builders[predicate[2]]
            )
        return BuilderLeafNode(predicate, arguments, builder)

    def build_modules_for_leaves(
        self, leaf_symbols: Iterable[Symbol]
    ) -> list[DeepLogModule]:
        """Build sub-modules in batch for a set of circuit-leaf symbols.

        Groups leaves by predicate and invokes each predicate's registered
        atom builder once on the full list of arg-tuples, returning one
        module per predicate. Leaves without a registered builder or with
        a non-standard shape are skipped — they become external inputs at
        composition time.
        """
        args_by_pred: dict[tuple[str, int, str], list[tuple]] = defaultdict(list)
        for sym in leaf_symbols:
            if len(sym) != 3 or sym[0] != "_":
                continue
            predicate, arguments = self._decode_leaf(sym)
            args_by_pred[predicate].append(arguments)

        modules: list[DeepLogModule] = []
        for predicate, args_list in args_by_pred.items():
            builder = self._atom_builders.get(predicate)
            if builder is None:
                continue
            modules.append(builder(args_list).to_module())
        return modules

    @staticmethod
    def _decode_leaf(atom: Symbol) -> tuple[tuple[str, int, str], tuple[Symbol, ...]]:
        """Split a leaf atom into its ``(functor, arity, structure)`` key and args."""
        structure_tag = cast(Symbol, atom[2])
        structure = structure_tag[0]
        literal = strip_literal_structure(cast(Symbol, atom[1]), structure)
        return (literal[0], len(literal) - 1, structure), literal[1:]


def to_module(
    *nodes: CompositeCircuitNode,
    names: Iterable[Symbol] | None = None,
    structure_override: str | None = None,
) -> DeepLogModule:
    """Convert one or more CompositeCircuitNodes into a DeepLogModule.

    All nodes must be from the same circuit. Each node's children modules are
    collected and composed with the circuit module.

    Args:
        *nodes: One or more CompositeCircuitNodes to use as roots.
        names: Optional output names for each root. If not provided,
               names are auto-generated from the circuit name.

    Returns:
        A DeepLogModule with the specified roots.
    """
    if not nodes:
        raise ValueError("At least one CompositeCircuitNode is required")

    circuit = nodes[0].root.circuit
    for n in nodes[1:]:
        if n.root.circuit is not circuit:
            raise ValueError("All CompositeCircuitNodes must be from the same circuit")

    names_list: list[Symbol]
    if names is None:
        names_list = [(f"{circuit.name}_{i}",) for i in range(len(nodes))]
    else:
        names_list = list(names)
        if len(names_list) != len(nodes):
            raise ValueError(f"Expected {len(nodes)} names, got {len(names_list)}")

    # Collect all child modules from all nodes
    all_children: list[SupportsToModule] = []
    for node in nodes:
        all_children.extend(node.children)

    # Build root spec
    root_spec = {n.root.node: name for n, name in zip(nodes, names_list, strict=True)}
    circuit_module = circuit.to_module(root_spec, structure_override=structure_override)

    all_modules = [circuit_module] + [child.to_module() for child in all_children]
    return compose_modules(
        all_modules,
        circuit_module.get_output_shape(),
        nodes[0].get_structure(),
    )

#  Copyright (c) 2024-2026. KU Leuven
"""Internal node types produced by :class:`DeepLogModuleFactory`.

All implement the :class:`~.builder_protocols.InternalNode` protocol
(``get_structure`` + ``to_module``). The factory builds these as it lowers a
symbolic formula; downstream they're either composed into a torch module
(via ``to_module()``) or used as inputs to a containing circuit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from ...algebraic import HasStructure
from ...circuit import Circuit
from ...circuit import CircuitNode
from ...module import DeepLogModule
from ...module import SupportsToModule
from ...module import compose_modules
from ...shape import Shape
from ...shape import map_shape
from ...symbol import Symbol
from ...symbol import with_structure
from .builder_protocols import AtomBuilder
from .builder_protocols import InternalNode


@dataclass
class AtomLeafNode(HasStructure):
    """A leaf node in a DeepLog formula representing an atom."""

    predicate: tuple[str, int, str]
    arguments: tuple[Symbol, ...]

    def get_structure(self) -> str:
        """Return the structure of the leaf."""
        return self.predicate[2]

    def __hash__(self):
        """Returns hash of predicate + arguments"""
        return hash(self.predicate + tuple(self.arguments))


@dataclass
class BuilderLeafNode(AtomLeafNode):
    """Atom leaf whose module is produced by a registered builder."""

    builder: AtomBuilder

    def to_module(self) -> DeepLogModule:
        """Build the module using the stored builder."""
        return self.builder([self.arguments]).to_module()


@dataclass
class InputLeafNode(AtomLeafNode):
    """Atom leaf with no builder; its circuit leaf name becomes a module input.

    Holds the structure's circuit builder so a leaf that ends up being the
    *whole* formula can still be compiled (see :meth:`to_module`).
    """

    circuit_builder: Callable[[], Circuit]

    def to_module(self) -> DeepLogModule:
        """Compile a bare leaf-only formula by wrapping it in a one-node circuit.

        When the leaf is combined with operators it is absorbed as a circuit
        input instead (see ``_ensure_circuit_node``), so this path runs only
        when the leaf is the entire formula. Building a single-node circuit and
        compiling it reuses the standard backends and keeps the degenerate
        cases on the circuit side: a variable leaf compiles to an identity
        (pass-through) over its one symbol, while a numeric symbol (e.g.
        ``0_fuzzy``) is resolved by the structure's ``constant_fn`` into a
        constant node and compiles to a constant module.
        """
        circuit = self.circuit_builder()
        leaf_symbol: Symbol = (
            "_",
            (self.predicate[0], *self.arguments),
            (self.predicate[2],),
        )
        node = circuit.get_leaf_node(leaf_symbol)
        return circuit.to_module({node: leaf_symbol})


@dataclass
class CompositeCircuitNode:
    """A circuit-node reference together with sub-modules that produce its leaf inputs.

    Composition over inheritance: ``root`` is a plain :class:`CircuitNode`
    (a reference into a circuit's graph); ``children`` are the predicate
    sub-modules that feed values into the circuit's leaves. The factory
    builds these as it lowers a symbolic formula, and ``to_module()``
    compiles the pair into a single :class:`DeepLogModule`.
    """

    root: CircuitNode
    children: list[SupportsToModule]

    def get_structure(self) -> str:
        """Return the structure of the underlying circuit node."""
        return self.root.get_structure()

    def to_module(
        self,
        name: Symbol | None = None,
        structure_override: str | None = None,
    ) -> DeepLogModule:
        """Compile the circuit node and compose it with the children modules."""
        circuit_module = self.root.to_module(
            name,
            structure_override=structure_override,
        )
        all_modules = [circuit_module] + [child.to_module() for child in self.children]
        structure = structure_override or self.get_structure()
        return compose_modules(
            all_modules,
            circuit_module.get_output_shape(),
            structure,
        )

    def __hash__(self):
        """Hash by the root reference; children are not part of identity."""
        return hash(self.root)


@dataclass
class ExpectationNode:
    """Lazy expectation node that defers circuit transformation until to_module().

    Stores the boolean circuit child and the leaf mapping so that multiple
    ExpectationNodes sharing the same circuit can be batch-transformed via
    ``transform_nodes`` before compilation.
    """

    child: CompositeCircuitNode
    leaf_mapping: Callable[[Symbol], Symbol]

    def get_structure(self) -> str:
        """Return the target structure for this expectation."""
        return "probability"

    def to_module(self):
        """Transform the boolean circuit to probability and compile to a module."""
        transformed = self.child.root.transform_circuit(
            "probability", leaf_mapping=self.leaf_mapping
        )
        module = transformed.to_module()
        module.structure = "probability"
        return module


def build_expectation(
    child: InternalNode,
    variables,
    params,
    domains,
) -> InternalNode:
    """Build an :class:`ExpectationNode` from a boolean circuit child.

    Leaves are tag-rewritten from boolean to probability via
    :func:`with_structure`. The compile pipeline overrides this default mapping
    with one derived from the engine's atom labels (see
    :func:`~deeplog.formula.distribution.build_leaf_mapping`).
    """
    if (
        not isinstance(child, CompositeCircuitNode)
        or child.get_structure() != "boolean"
    ):
        raise ValueError(
            "Expectation currently only supports boolean circuit children."
        )

    if params:
        raise ValueError(
            "Expectation does not accept probability-formula parameters; "
            "the boolean-to-probability leaf mapping is built from atom labels."
        )

    leaf_mapping = partial(with_structure, structure="probability")
    return ExpectationNode(child, leaf_mapping)


# Elementwise functions that cast a tensor from one algebraic structure to
# another. Extend this table to support a new (from, to) pair; the keys are
# matched against the strings passed to register_transformation_builder.
_CAST_FUNCTIONS: dict[tuple[str, str], Callable[[torch.Tensor], torch.Tensor]] = {
    ("boolean", "probability"): lambda x: x,
    ("real", "probability"): torch.sigmoid,
}


class _StructureCast(DeepLogModule):
    """Apply a prepared elementwise function and report the target structure."""

    def __init__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        input_shape: Shape,
        output_shape: Shape,
        structure: str,
    ):
        """Bind the elementwise function and the structure to report."""
        super().__init__(input_shape, output_shape)
        self._func = func
        self._structure = structure

    def forward(self, *inputs):
        """Apply the elementwise function to one or more tensors."""
        if len(inputs) == 1:
            return self._func(inputs[0])
        return tuple(self._func(x) for x in inputs)

    def get_structure(self) -> str:
        """Return the target algebraic structure of this cast."""
        return self._structure


def build_transform(
    input_shape: Shape, from_structure: str, to_structure: str
) -> _StructureCast:
    """Build the registered elementwise cast from ``from_structure`` to ``to_structure``.

    Looks up the cast in :data:`_CAST_FUNCTIONS`, re-tags the symbolic
    output shape with ``to_structure``, and wraps both in a
    :class:`_StructureCast` module.
    """
    func = _CAST_FUNCTIONS[(from_structure, to_structure)]
    output_shape = map_shape(
        lambda symbol: ("_", ("transform", (to_structure,), symbol), (to_structure,)),
        input_shape,
    )
    return _StructureCast(func, input_shape, output_shape, to_structure)

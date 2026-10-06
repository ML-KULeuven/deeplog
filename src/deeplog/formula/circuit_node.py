#  Copyright (c) 2024-2026. KU Leuven
"""Circuit-node helpers.

A ``CircuitNode`` is plain, frozen AST data (see :mod:`deeplog.formula.ast`).
Operating on a lump is a free function: :func:`to_module` / :func:`transform_nodes`
to compile or transform it, and ``node.circuit.structure`` to read its structure.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import cast

from torch import Tensor

from ..algebraic import AlgebraicStructure
from ..circuit.circuit import Circuit
from ..circuit.transform import transform_circuit
from ..module.columnwise import ColumnwiseModule
from ..module.deeplog_module import DeepLogModule
from ..symbol import Symbol
from ..symbol import with_structure
from .ast import CircuitNode


def lump_name(node: CircuitNode) -> Symbol:
    """The name a lump's value is compiled under: its node id in its circuit.

    Labelled with the circuit's algebra, as
    :func:`~deeplog.circuit.lower.dispatch.to_module` labels every root it
    compiles.
    """
    return with_structure((f"{node.circuit.name}_n{node.node}",), node.structure)


def cast_name(structure: str, source: Symbol) -> Symbol:
    """The name of the value named ``source``, cast into ``structure``.

    Construction names a cast's leaf with it and lowering names the cast's
    column with it, so the column feeds the leaf.
    """
    return with_structure(("transform", (structure,), source), structure)


def _columns(columns: Tensor) -> Tensor:
    """``columns`` as they are."""
    return columns


def _single_circuit(nodes: tuple[CircuitNode, ...]) -> Circuit:
    """Return the one circuit shared by ``nodes``, or raise if empty / mixed."""
    if not nodes:
        raise ValueError("At least one CircuitNode is required")
    circuit = nodes[0].circuit
    for n in nodes[1:]:
        if n.circuit is not circuit:
            raise ValueError("All CircuitNodes must be from the same circuit")
    return circuit


def transform_nodes(
    *nodes: CircuitNode,
    target_structure: str | AlgebraicStructure,
    operator_mapping: dict[str, str] | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
) -> tuple[CircuitNode, ...]:
    """Transform one or more co-resident ``CircuitNode`` roots."""
    circuit = _single_circuit(nodes)
    new_circuit, node_map = transform_circuit(
        circuit,
        target_structure,
        [n.node for n in nodes],
        operator_mapping=operator_mapping,
        leaf_mapping=leaf_mapping,
    )
    return tuple(CircuitNode(new_circuit, node_map[n.node]) for n in nodes)


def to_module(
    *nodes: CircuitNode | Mapping[Symbol, CircuitNode],
    names: tuple[Symbol, ...] | None = None,
) -> DeepLogModule:
    """Convert one or more co-resident ``CircuitNode`` roots into a module.

    Roots are given either as positional ``CircuitNode`` arguments — named
    positionally via ``names``, or ``<circuit>_<i>`` by default — or as a
    single ``{name: node}`` mapping, the shape multi-root producers (grounder
    proofs, engine result formulas) already have. The mapping form forbids
    ``names``; insertion order fixes the output order, exactly as the
    positional form does.

    Two roots may be the same node: knowledge compilation is canonical, so
    equivalent formulas compile to one. Each name still gets its own output
    column, selected from the node that was compiled.
    """
    if len(nodes) == 1 and isinstance(nodes[0], Mapping):
        if names is not None:
            raise ValueError("names must be omitted when roots are a mapping.")
        names = tuple(nodes[0].keys())
        nodes = tuple(nodes[0].values())
    elif any(isinstance(n, Mapping) for n in nodes):
        raise ValueError("A roots mapping must be the only positional argument.")
    node_roots = cast(tuple[CircuitNode, ...], nodes)

    circuit = _single_circuit(node_roots)
    if names is None:
        names = tuple((f"{circuit.name}_{i}",) for i in range(len(node_roots)))
    elif len(names) != len(node_roots):
        raise ValueError(f"Expected {len(node_roots)} names, got {len(names)}")

    roots: dict[int, Symbol] = {}
    for n, name in zip(node_roots, names, strict=True):
        roots.setdefault(n.node, name)
    module = circuit.to_module(roots)
    if len(roots) == len(node_roots):
        return module

    # A node named more than once is compiled once and read twice. Both columns
    # carry the compiled algebra's label, the way ``to_module`` labels the roots
    # it did compile.
    structure = circuit.structure.name
    compiled = tuple(with_structure(roots[n.node], structure) for n in node_roots)
    return ColumnwiseModule(
        _columns,
        module,
        compiled,
        names=tuple(with_structure(name, structure) for name in names),
        name="roots",
    )

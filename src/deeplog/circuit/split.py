#  Copyright (c) 2024-2026. KU Leuven
"""Lower a circuit across the operators its backend has no node for.

A backend builds its graph out of a semiring's product, sum and complement, so
an operator that is none of those — a semifield's ``divide`` — has no node. The
graph is *cut* at such a node: the operand subgraphs are lowered as roots of
one ordinary lowering, the algebra's own ``operator_fns`` entry combines
their output columns, and the resulting value is handed to the circuit above the
cut as an ordinary leaf.

The region above the cuts is the same circuit lowered again, *bounded* at them:
each cut is an input slot named for the value that feeds it (``frontier``, down
to :meth:`~deeplog.circuit.graph.Graph.iter_topological`). The operands of the
topmost cuts go through the ordinary entry point, which cuts *them* in turn, so
a division under a division needs no extra machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..module import ColumnwiseModule
from ..module import compose_modules
from ..module import reshape
from ..shape import SymTensor
from ..shape import get_all_symbols
from ..symbol import Symbol
from ..symbol import with_structure
from .circuit import Circuit


if TYPE_CHECKING:
    from ..module import DeepLogModule


def find_cuts(
    circuit: Circuit,
    root_ids: list[int],
    routed: frozenset[str],
    frontier: Mapping[int, Symbol] | None = None,
) -> set[int]:
    """Operator nodes reachable from ``root_ids`` that ``routed`` does not cover.

    ``routed`` is :func:`~deeplog.circuit.backends.routed_operators` for the
    backend lowering this circuit. Only operator nodes can be cut: leaves,
    constants and nodes already on ``frontier`` are values the backend seeds
    itself.
    """
    operators = circuit.structure.operators
    return {
        node_id
        for node_id in circuit.iter_topological(root_ids, frontier)
        if node_id not in (frontier or ())
        and (node_type := circuit._get_node(node_id).node_type) in operators
        and node_type not in routed
    }


def lower_across_cuts(
    circuit: Circuit,
    roots: dict[int, Symbol],
    cuts: set[int],
    *,
    frontier: Mapping[int, Symbol] | None = None,
) -> DeepLogModule:
    """Lower ``circuit`` in partitions separated by ``cuts``.

    One lowering for the operands of the topmost cuts, one
    :class:`~deeplog.module.ColumnwiseModule` per operator applying the algebra's
    function to their columns, and one lowering of the region above, bounded
    at the cuts. When every root is itself a cut there is nothing above.
    """
    from .lower.dispatch import to_module

    root_ids = list(roots)
    above = _above_cuts(circuit, root_ids, cuts | set(frontier or ()))
    # Topological order, so the boundary (and every column group derived from it)
    # is deterministic.
    boundary = [
        node_id
        for node_id in circuit.iter_topological(root_ids)
        if node_id in cuts and node_id in above
    ]

    # The operands of every topmost cut are roots of *one* lowering, so the
    # work they share is shared, and a deeper cut among them is split there.
    operands: list[int] = []
    for node_id in boundary:
        for child in circuit._get_node(node_id).children:
            if child not in operands:
                operands.append(child)
    lower = to_module(
        circuit,
        {node_id: _value_symbol(circuit, node_id) for node_id in operands},
        frontier=frontier,
    )

    # A cut that the caller asked for by name is *named* by it: the value the
    # combination computes is that output, and the circuit above (if any) reads
    # it under the same name. Anything else is named for the node it computes.
    def cut_name(node_id: int) -> Symbol:
        return roots.get(node_id) or _value_symbol(circuit, node_id)

    combined = [
        _combine(circuit, lower, members, operator, arity, cut_name)
        for (operator, arity), members in _by_operator(circuit, boundary).items()
    ]
    wanted = SymTensor([roots[node_id] for node_id in root_ids])

    # A root that is itself a cut is already computed, so only the rest go
    # through a circuit — asking for one as both a root and a cut leaf would
    # make it its own input.
    boundary_set = set(boundary)
    upper_roots = {
        node_id: name for node_id, name in roots.items() if node_id not in boundary_set
    }
    if not upper_roots:
        return _packed(
            circuit,
            root_ids,
            reshape(compose_modules(combined, wanted), output=wanted),
            frontier,
        )

    # The region above the cuts is this same circuit, lowered bounded at them:
    # each cut is an input slot named for the value the combination computes, so
    # the two are wired by symbol like any other boundary.
    upper = to_module(
        circuit,
        upper_roots,
        frontier={**(frontier or {}), **{n: cut_name(n) for n in boundary}},
    )
    return _packed(
        circuit, root_ids, compose_modules([upper, *combined], wanted), frontier
    )


def _packed(
    circuit: Circuit,
    root_ids: list[int],
    module: DeepLogModule,
    frontier: Mapping[int, Symbol] | None = None,
) -> DeepLogModule:
    """Present the composition as one packed input tensor, like an uncut lowering.

    Composing partitions yields one input channel per symbol, but a lowered
    circuit's contract is a single tensor whose columns are its reachable
    leaves. Ordered by the circuit's own leaf order, so the interface does not
    depend on where the cuts fell; a symbol that is not a leaf of this circuit
    (an MV-SDD padding slot) keeps a stable place after them.
    """
    needed = set(get_all_symbols(module.get_input_shape()))
    order = [
        name for name in circuit.reachable_leaves(root_ids, frontier) if name in needed
    ]
    order += sorted(needed - set(order))
    return reshape(module, input=SymTensor(order))


def _value_symbol(circuit: Circuit, node_id: int) -> Symbol:
    """The name of the value node ``node_id`` computes.

    Keyed by node id rather than descriptive, since circuits deduplicate leaves
    but not operators: two distinct nodes can apply the same operator to the same
    operands, and these names have to tell their columns apart. Spelled and
    labelled the way a lump's root is at a boundary
    (:meth:`~deeplog.formula.circuit_factory.CircuitFactory.create_transformation`).
    """
    return with_structure((f"{circuit.name}_n{node_id}",), circuit.structure.name)


def _above_cuts(circuit: Circuit, root_ids: list[int], cuts: set[int]) -> set[int]:
    """Nodes reachable from ``root_ids`` without descending through a cut.

    The cuts themselves are included: they bound the upper region, where they
    stand for the value fed in rather than for the operator that produced it. A
    node both above and below a cut appears in both partitions, and the two agree
    because a leaf is named by its symbol.
    """
    seen: set[int] = set()
    stack = list(root_ids)
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id in cuts:
            continue
        stack.extend(circuit._get_node(node_id).children)
    return seen


def _by_operator(
    circuit: Circuit, boundary: list[int]
) -> dict[tuple[str, int], list[int]]:
    """Group cuts by the operator and arity they apply, preserving order."""
    groups: dict[tuple[str, int], list[int]] = {}
    for node_id in boundary:
        node = circuit._get_node(node_id)
        groups.setdefault((node.node_type, len(node.children)), []).append(node_id)
    return groups


def _combine(
    circuit: Circuit,
    lower: DeepLogModule,
    members: list[int],
    operator: str,
    arity: int,
    cut_name: Callable[[int], Symbol],
) -> DeepLogModule:
    """Apply ``operator`` to the operand columns of every cut in ``members``.

    One module for the whole group: ``lower`` is evaluated once and the operator
    applied across one column group per operand position. A circuit with two
    different unroutable operators gets one of these per operator.
    """
    operator_fn = circuit.structure.get_operator_fn(operator)
    if operator_fn is None:
        raise ValueError(
            f"Structure '{circuit.structure.name}' has a '{operator}' node but no "
            f"'{operator}' in operator_fns, so the value it computes has no meaning."
        )
    columns = tuple(
        tuple(
            _value_symbol(circuit, circuit._get_node(node_id).children[position])
            for node_id in members
        )
        for position in range(arity)
    )
    names = tuple(cut_name(node_id) for node_id in members)
    return ColumnwiseModule(operator_fn, lower, *columns, names=names, name=operator)

#  Copyright (c) 2024-2026. KU Leuven
"""Re-emit a circuit through a formula algebra — the circuit's catamorphism.

A circuit node is an atom, a unary or a binary: the fragment of the formula
language a circuit can express. Leaves, named constants and arbitrary numeric
constants are all atoms (:meth:`~deeplog.circuit.circuit.Circuit.get_symbol_name`
gives each one symbol), and every operator node is unary or binary, because every
one is built through
:meth:`~deeplog.circuit.circuit.Circuit.get_operator` with one or two children.

So a circuit is a sub-algebra of the formula AST, and folding one needs no
interface of its own: the same
:class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory` that
:func:`deeplog.formula.ast.fold` drives over a tree drives this over a graph.
An algebra that only ever meets circuit nodes raises on the three eliminators a
circuit has no node for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..formula.deeplogformulafactory import DeepLogFormulaFactory
    from ..symbol import Symbol
    from .circuit import Circuit


def fold_circuit[T](
    circuit: Circuit,
    roots: list[int],
    algebra: DeepLogFormulaFactory[T],
    *,
    frontier: Mapping[int, Symbol] | None = None,
    memo: dict[int, T] | None = None,
) -> dict[int, T]:
    """Re-emit the subgraph under ``roots`` through ``algebra``.

    Returns the carrier built for every node reached, keyed by source node id --
    so ``result[root]`` is the root's, and the map itself is the record of what
    each node became.

    Children are visited before parents, so an algebra may be stateful and see
    its ``create_*`` calls in dependency order. A node reached twice is built
    once: circuit ids are canonical, so unlike the AST fold
    (:func:`deeplog.formula.ast.fold`, which relies on hash-consing) memoising by
    id needs nothing established first.

    ``frontier`` nodes are not descended into and become atoms under the symbols
    given, which is how a bounded lowering treats a cut
    (:mod:`deeplog.circuit.split`). ``memo`` seeds the result and is mutated in
    place, so successive folds sharing one algebra build a subgraph once between
    them -- as :func:`deeplog.formula.ast.fold`'s does on the AST side.
    """
    built: dict[int, T] = memo if memo is not None else {}
    for node_id in circuit.iter_topological(roots, frontier):
        if node_id in built:
            continue
        if frontier is not None and node_id in frontier:
            built[node_id] = algebra.create_atom(frontier[node_id])
            continue
        symbol = circuit.get_symbol_name(node_id)
        if symbol is not None:
            built[node_id] = algebra.create_atom(symbol)
            continue
        node = circuit._get_node(node_id)
        children = [built[child] for child in node.children]
        if len(children) == 1:
            built[node_id] = algebra.create_unary_node(node.node_type, children[0])
        elif len(children) == 2:
            built[node_id] = algebra.create_binary_node(
                node.node_type, children[0], children[1]
            )
        else:
            raise ValueError(
                f"Circuit node {node_id} applies '{node.node_type}' to "
                f"{len(children)} operands; a circuit's operators are unary or "
                f"binary, which is what the formula algebra has eliminators for."
            )
    return built

#  Copyright (c) 2024-2026. KU Leuven
"""Execution-strategy rewrites for recognized aggregations.

Recognition (:mod:`deeplog.formula.passes`) labels *what* a sub-formula means;
a strategy decides *how* it is computed and is free to be swapped or gated. The
weighted-model-count fast path is one strategy for an ``expectation``: instead of
enumerating the binder domains, it knowledge-compiles the boolean circuit and
transforms the result into probability.

This module owns that strategy end to end — which aggregations the construction
fold may absorb (:func:`absorb_aggregation`), the boundary name they meet the
lowering at (:func:`expectation_symbol`), and the count itself
(:func:`transform_expectation_to_probability`).
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from functools import partial

from ..circuit.circuit import Circuit
from ..circuit.knowledge_compile import knowledge_compile
from ..symbol import Symbol
from ..symbol import with_structure
from ..variable import VariableAtoms
from .ast import Aggregation
from .ast import CircuitNode
from .ast import FormulaNode
from .circuit_node import lump_name


def expectation_symbol(child: CircuitNode) -> Symbol:
    """The probability-valued name of the weighted model count of ``child``.

    Minted in one place because two sides have to agree on it exactly: the leaf
    :func:`absorb_aggregation` places in the probability circuit, and the output
    column the lowered count carries. If they drift nothing raises — the leaf becomes an external input
    and the count is computed and discarded.
    """
    lump = with_structure(lump_name(child), child.circuit.structure.name)
    return with_structure(("expectation", lump), "probability")


def absorb_aggregation(
    get_circuit: Callable[[str], Circuit],
    operation: str,
    binders: Sequence[Symbol],
    params: Sequence[FormulaNode],
    child: FormulaNode,
) -> CircuitNode | None:
    """The lump a circuit-representable aggregation collapses to, or ``None``.

    The construction fold offers every aggregation here and keeps its symbolic
    node when the answer is ``None``, so which aggregations have a circuit form
    is decided by this module rather than by the fold.

    ``expectation`` is the one that does. Its count is *deferred*, exactly as a
    cross-structure cast is: a leaf in the probability circuit obtained from
    ``get_circuit``, fed by the boolean lump it counts. Knowledge compilation is
    a compile step, and running it here — once per expectation, as the fold
    reaches it — would both compile during AST construction and give up the
    sharing between counts over one circuit. ``binders`` are dropped: the count
    is over every model of the circuit, which is what enumerating them yields.
    """
    if operation != "expectation" or not isinstance(child, CircuitNode):
        return None
    if child.circuit.structure.name != "boolean":
        raise ValueError(
            "Expectation currently only supports boolean circuit children."
        )
    if params:
        raise ValueError(
            "Expectation does not accept probability-formula parameters; "
            "the boolean-to-probability leaf mapping is built from atom labels."
        )
    if child.feeders:
        # A boolean lump whose leaves are fed by cross-structure casts cannot
        # be model-counted: the leaf-symbol rewrite would detach the feeder.
        raise ValueError(
            "Expectation over a boolean circuit with cross-structure boundary "
            "feeders is not supported."
        )
    target = get_circuit("probability")
    leaf_name = expectation_symbol(child)
    return CircuitNode(
        target,
        target.get_leaf_node(leaf_name),
        ((leaf_name, Aggregation("expectation", (), (), child)),),
    )


def deferred_lump(node: FormulaNode) -> CircuitNode | None:
    """The boolean lump whose count ``node`` defers, or ``None``.

    Reads back what :func:`absorb_aggregation` minted: the aggregation it records
    as a leaf's feeder, with the binders it dropped. An ``expectation`` that
    still carries binders has not been absorbed and is not one of these.
    """
    if (
        isinstance(node, Aggregation)
        and node.operation == "expectation"
        and not node.binders
        and not node.params
        and isinstance(node.child, CircuitNode)
    ):
        return node.child
    return None


def transform_expectation_to_probability(
    *boolean_nodes: CircuitNode,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    variables: VariableAtoms | None = None,
) -> tuple[CircuitNode, ...]:
    """Turn co-resident boolean circuit roots into their weighted model counts.

    :func:`~deeplog.circuit.knowledge_compile.knowledge_compile` rewrites the
    formula into a deterministic, decomposable equivalent and emits it straight
    into the probability semiring, reading its roles there. Two steps in one:
    the rewrite preserves roles, so reading them in another algebra is exact
    once — and only once — the formula is a d-DNNF, which is what makes the
    single call safe where a transform of the *uncompiled* circuit would not be.

    ``leaf_mapping`` ``None`` rewrites each boolean leaf tag to probability;
    DeepProbLog compilation passes a label-derived mapping instead, under which a
    numeric label folds to a constant node. ``variables`` says where each
    multi-valued variable occurs.

    All roots must share one boolean circuit and are compiled in a single batch,
    so the results co-reside in one probability circuit.
    """
    if not boolean_nodes:
        raise ValueError("At least one boolean node is required.")
    circuit = boolean_nodes[0].circuit
    for node in boolean_nodes[1:]:
        if node.circuit is not circuit:
            raise ValueError("All CircuitNodes must be from the same circuit")
    if leaf_mapping is None:
        leaf_mapping = partial(with_structure, structure="probability")

    roots = [node.node for node in boolean_nodes]
    counted, counted_map = knowledge_compile(
        circuit,
        roots,
        variables=variables,
        structure="probability",
        leaf_mapping=leaf_mapping,
    )
    return tuple(CircuitNode(counted, counted_map[root]) for root in roots)

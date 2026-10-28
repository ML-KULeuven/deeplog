#  Copyright (c) 2024-2026. KU Leuven
"""Reductions: aggregations computed by reducing their operands over their binders' values."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...algebraic import AggregationFn
from ...algebraic import AlgebraicStructure
from ...algebraic import structure_registry
from ...module.aggregation_modules import AggregationModule
from ...module.deeplog_module import DeepLogModule
from ...symbol import Symbol
from ...symbol import symbol_to_str
from ..ast import Aggregation
from ..ast import AggregationOp


if TYPE_CHECKING:
    from .compiler import Compiler
    from .lowering import Lowering


def reduction(
    nodes: Sequence[Aggregation], lowering: Lowering, /
) -> list[tuple[DeepLogModule, int]]:
    """Compute each aggregation by reducing its operands over its binders' values.

    The operands of the aggregations over the same binders are lowered
    together, each with an axis for every one of those binders free in it,
    which the aggregation's algebra reduces with its aggregator for the
    aggregation's operation (:data:`~deeplog.algebraic.AggregationFn`). So each
    subformula is evaluated once per assignment of the binders free in it, and
    aggregations over the same binders evaluate what they share once. An atom
    that holds binders at arguments of sort a named or a tensor domain is read
    from its ground atoms, one for each combination of those binders' values,
    which its builder receives in its place; a binder whose values cannot be
    written, as rows of a tensor that are not numbers, is not among them. A
    compiler computes this way every operation it has no other builder for.

    Raises:
        NotImplementedError: If an aggregation's algebra has no aggregator for
            its operation.
    """
    reports: dict[int, tuple[DeepLogModule, int]] = {}
    groups: dict[tuple[Symbol, ...], list[int]] = {}
    for index, node in enumerate(nodes):
        groups.setdefault(node.binders, []).append(index)
    for binders, members in groups.items():
        operands = {
            index: (nodes[index].child, *nodes[index].params) for index in members
        }
        distinct = list(dict.fromkeys(o for index in members for o in operands[index]))
        body = lowering._lower_over(binders, distinct)
        reductions: dict[tuple[str, tuple[Symbol, ...]], AggregationFn] = {}
        keys = {}
        for index in members:
            read = tuple(lowering._column(operand) for operand in operands[index])
            keys[index] = (nodes[index].operation, read)
            reductions.setdefault(
                keys[index], aggregator(nodes[index], lowering.compiler)
            )
        module = AggregationModule(
            [(binder, len(lowering.compiler.domain(binder))) for binder in binders],
            body,
            [(operation, op, read) for (operation, read), op in reductions.items()],
        )
        order = list(reductions)
        for index in members:
            reports[index] = (module, order.index(keys[index]))
    return [reports[index] for index in range(len(nodes))]


def aggregator(node: Aggregation | AggregationOp, compiler: Compiler) -> AggregationFn:
    """The aggregator of ``node``'s algebra for its operation.

    Raises:
        NotImplementedError: If the algebra has none.
    """
    if node.structure is None:
        over = node.child if isinstance(node, Aggregation) else node.operands
        raise NotImplementedError(
            f"The {node.operation!r} aggregation over {over!r} is a value "
            "of no algebra: its formula gives it none."
        )
    algebra = compiler.algebra(node.structure)
    found = algebra.get_aggregation_fn(node.operation)
    if found is None:
        raise NotImplementedError(
            f"{algebra.name!r} has no aggregator {node.operation!r}; its "
            f"aggregators are {sorted(algebra.aggregations)}. "
            f"{_instead(node, algebra, compiler)}"
        )
    return found


def _instead(
    node: Aggregation | AggregationOp, algebra: AlgebraicStructure, compiler: Compiler
) -> str:
    """What to write for ``node``, whose ``algebra`` has no aggregator for its operation.

    The algebras it suggests have that aggregator and a cast from ``algebra``,
    among the registered ones and the compiler's own.
    """
    operation = node.operation
    algebras = {**structure_registry, **compiler._structures}
    targets = sorted(
        name
        for name, other in algebras.items()
        if operation in other.aggregations
        and (algebra.name, name) in compiler._transformation_builders
    )
    if not targets:
        return (
            f"No algebra that {algebra.name!r} casts into has {operation!r} either; "
            f"an algebra declares its aggregators as its aggregation_fns."
        )
    if isinstance(node, AggregationOp):
        written = " or ".join(f"{operation}[(φ)_{name}, ...]" for name in targets)
        return f"Cast each operand φ into an algebra that has it: {written}."
    binders = ", ".join(map(symbol_to_str, node.binders))
    written = " or ".join(f"{operation}({binders}): (φ)_{name}" for name in targets)
    return f"Cast the body φ into an algebra that has it: {written}."

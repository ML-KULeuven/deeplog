#  Copyright (c) 2024-2026. KU Leuven
"""Reductions: aggregations computed by enumerating their binders' domains."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

from torch import Tensor

from ...algebraic import AggregationFn
from ...algebraic import AlgebraicStructure
from ...algebraic import structure_registry
from ...module.aggregation_modules import AggregationModule
from ...module.deeplog_module import DeepLogModule
from ...symbol import symbol_to_str
from ..ast import Aggregation


if TYPE_CHECKING:
    from .compiler import Compiler
    from .lowering import Lowering


def reduction(
    nodes: Sequence[Aggregation], lowering: Lowering, /
) -> list[tuple[DeepLogModule, int]]:
    """Compute each aggregation from its operands' values at every assignment of its binders.

    The body and the params are evaluated at every assignment, and the
    aggregation's algebra aggregates them with its aggregator for the
    aggregation's operation (:data:`~deeplog.algebraic.AggregationFn`). A
    compiler computes this way every operation it has no other builder for.

    Raises:
        NotImplementedError: If an aggregation's algebra has no aggregator for
            its operation.
    """
    return [(_reduce(node, lowering), 0) for node in nodes]


def aggregator(node: Aggregation, compiler: Compiler) -> AggregationFn:
    """The aggregator of ``node``'s algebra for its operation.

    Raises:
        NotImplementedError: If the algebra has none.
    """
    if node.structure is None:
        raise NotImplementedError(
            f"The {node.operation!r} aggregation over {node.child!r} is a value "
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


def _instead(node: Aggregation, algebra: AlgebraicStructure, compiler: Compiler) -> str:
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
    binders = ", ".join(map(symbol_to_str, node.binders))
    written = " or ".join(f"{operation}({binders}): (φ)_{name}" for name in targets)
    return f"Cast the body φ into an algebra that has it: {written}."


def _reordered(op: AggregationFn, order: tuple[int, ...], *columns: Tensor) -> Tensor:
    """``op`` of the ``columns`` at ``order``."""
    return op(*(columns[i] for i in order))


def _reduce(node: Aggregation, lowering: Lowering) -> AggregationModule:
    """The module aggregating ``node`` with its algebra's aggregator.

    Operands that are one formula are lowered to one column, which the
    aggregator receives in each of their places.
    """
    op = aggregator(node, lowering.compiler)
    operands = (node.child, *node.params)
    positions = {operand: i for i, operand in enumerate(dict.fromkeys(operands))}
    order = [positions[operand] for operand in operands]
    return AggregationModule(
        lowering.lower(*positions),
        list(node.binders),
        [lowering.compiler.domain(binder).as_tensor() for binder in node.binders],
        name=node.operation,
        op=partial(_reordered, op, tuple(order)),
    )

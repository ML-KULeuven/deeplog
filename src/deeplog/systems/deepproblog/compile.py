#  Copyright (c) 2024-2026. KU Leuven
"""Compile DeepProbLog engine results into DeepLogModules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...circuit import to_module as circuit_to_module
from ...circuit import transform_nodes
from ...formula.deeplogmodulefactory.nodes import ExpectationNode
from ...formula.distribution import build_probability_distribution
from ...module.module_circuit import compose_modules
from .engine import EngineResult


if TYPE_CHECKING:
    from ...formula.deeplogmodulefactory import DeepLogModuleFactory
    from ...module import DeepLogModule


def compile_to_module(
    result: EngineResult,
    factory: DeepLogModuleFactory,
) -> DeepLogModule:
    """Compile an engine result to a DeepLogModule.

    Builds an expectation aggregation for each formula in the result,
    batch-transforms the shared boolean circuit to the probability semiring
    in a single pass, and composes with predicate modules.

    Args:
        result: The engine result containing formulas and labels.
        factory: The module factory to use.

    Returns:
        A composed DeepLogModule.
    """
    prob_dist = build_probability_distribution(result.labels, factory)
    params = [prob_dist] if prob_dist is not None else []

    answers = tuple(result.formulas.keys())
    exp_nodes: list[ExpectationNode] = []
    for formula in result.formulas.values():
        node = factory.create_aggregation("expectation", [], params, formula)
        if not isinstance(node, ExpectationNode):
            raise TypeError("Expected all aggregation results to be ExpectationNodes.")
        exp_nodes.append(node)

    # Batch-transform all children at once (single circuit transform)
    leaf_mapping = exp_nodes[0].leaf_mapping
    roots = [n.child.root for n in exp_nodes]
    transformed = transform_nodes(
        *roots,
        target_structure="probability",
        leaf_mapping=leaf_mapping,
    )

    raw_module = circuit_to_module(*transformed, names=answers)
    child_modules = factory.build_modules_for_leaves(transformed[0].circuit.leaf_nodes)

    if not child_modules:
        return raw_module
    return compose_modules(
        child_modules + [raw_module],
        raw_module.get_output_shape(),
        "probability",
    )

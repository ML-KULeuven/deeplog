#  Copyright (c) 2024-2026. KU Leuven
"""Compile DeepProbLog engine results into DeepLogModules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...circuit import to_module as circuit_to_module
from ...circuit import transform_nodes
from ...formula.deeplogmodulefactory.nodes import ExpectationNode
from ...formula.distribution import build_leaf_mapping
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
    # Map each labeled boolean atom to its probability label directly from the
    # engine labels — unambiguous even when distinct atoms share arguments.
    leaf_mapping = build_leaf_mapping(result.labels)

    answers = tuple(result.formulas.keys())
    exp_nodes: list[ExpectationNode] = []
    for formula in result.formulas.values():
        node = factory.create_aggregation("expectation", [], [], formula)
        if not isinstance(node, ExpectationNode):
            raise TypeError("Expected all aggregation results to be ExpectationNodes.")
        # Replace the default tag-rewrite mapping with the label-derived one.
        node.leaf_mapping = leaf_mapping
        exp_nodes.append(node)

    # Batch-transform all children at once (single circuit transform)
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

#  Copyright (c) 2024-2026. KU Leuven
"""Default builder registrations populated into the :mod:`.registry`."""

from __future__ import annotations

from functools import partial

from ...circuit.circuit import Circuit
from ...module import AggregationModule
from ..predicates import EqualityPredicate
from ..predicates.builtin_predicates import ProbabilityPredicate
from .nodes import build_expectation
from .nodes import build_transform
from .registry import register_aggregation_builder
from .registry import register_atom_builder
from .registry import register_circuit_builder
from .registry import register_transformation_builder


def register_defaults() -> None:
    """Populate the registry with deeplog's default builders.

    Called explicitly from the package ``__init__`` at import time so the
    registration is visible (not an import-side-effect on this module).
    """
    register_atom_builder(
        *EqualityPredicate.get_predicate(),
        partial(EqualityPredicate, domain=[("false",), ("true",)]),
    )
    register_atom_builder(*ProbabilityPredicate.get_predicate(), ProbabilityPredicate)

    register_circuit_builder("boolean", lambda: Circuit("boolean", deterministic=True))
    register_circuit_builder("probability", lambda: Circuit("probability"))
    register_circuit_builder("logprobability", lambda: Circuit("logprobability"))

    register_aggregation_builder(
        "sum",
        lambda child, variables, _params, domains: AggregationModule(
            child, variables, domains, name="sum", op=lambda x: x.sum(dim=1)
        ),
    )
    register_aggregation_builder("expectation", build_expectation)

    register_transformation_builder(
        "boolean",
        "probability",
        partial(build_transform, from_structure="boolean", to_structure="probability"),
    )

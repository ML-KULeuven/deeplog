#  Copyright (c) 2024-2026. KU Leuven
"""Default builder registrations populated into the :mod:`.registry`."""

from __future__ import annotations

from functools import partial

from ...circuit.circuit import Circuit
from ...module import AggregationModule
from ..predicates import EqualityPredicate
from ..predicates.builtin_predicates import LogProbabilityPredicate
from ..predicates.builtin_predicates import ProbabilityPredicate
from .registry import register_aggregation_builder
from .registry import register_atom_builder
from .registry import register_circuit_builder
from .registry import register_transformation_builder
from .transform_cast import build_transform


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
    register_atom_builder(
        *LogProbabilityPredicate.get_predicate(), LogProbabilityPredicate
    )

    register_circuit_builder("boolean", lambda: Circuit("boolean"))
    register_circuit_builder("probability", lambda: Circuit("probability"))
    register_circuit_builder("logprobability", lambda: Circuit("logprobability"))

    register_aggregation_builder(
        "sum",
        lambda child, variables, _params, domains: AggregationModule(
            child, variables, domains, name="sum", op=lambda x: x.sum(dim=1)
        ),
    )

    for source, target in (
        ("boolean", "probability"),
        ("probability", "logprobability"),
        ("logprobability", "probability"),
    ):
        register_transformation_builder(
            source,
            target,
            partial(build_transform, from_structure=source, to_structure=target),
        )

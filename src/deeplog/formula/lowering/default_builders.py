#  Copyright (c) 2024-2026. KU Leuven
"""The builders every compiler starts from, by the key it looks them up under.

A :class:`~deeplog.formula.lowering.compiler.Compiler`'s own builders win over
these, key by key.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from ..predicates.builtin_predicates import LogProbabilityPredicate
from ..predicates.builtin_predicates import ProbabilityPredicate
from .builder_protocols import AggregationBuilder
from .builder_protocols import AtomBuilder
from .builder_protocols import TransformationBuilder
from .expectation import weighted_model_count
from .transform_cast import build_transform


#: The atom builders, keyed ``(functor, arity, structure)``.
default_atom_builders: Mapping[tuple[str, int, str], AtomBuilder] = {
    ("p", 2, "probability"): ProbabilityPredicate,
    ("logp", 2, "logprobability"): LogProbabilityPredicate,
}


#: The aggregation builders, keyed by operation;
#: :func:`~deeplog.formula.lowering.reduction.reduction` builds every other.
default_aggregation_builders: Mapping[str, AggregationBuilder] = {
    "expectation": weighted_model_count,
}

#: The casts between algebras, keyed ``(from, to)``.
default_transformation_builders: Mapping[tuple[str, str], TransformationBuilder] = {
    (source, target): partial(
        build_transform, from_structure=source, to_structure=target
    )
    for source, target in (
        ("boolean", "probability"),
        ("boolean", "real"),
        ("boolean", "logprobability"),
        ("boolean", "mpe"),
        ("probability", "logprobability"),
        ("logprobability", "probability"),
    )
}

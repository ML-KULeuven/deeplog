#  Copyright (c) 2024-2026. KU Leuven
"""Builder registries consulted by :class:`DeepLogModuleFactory`.

Module-level dicts seed each new factory instance; ``register_*`` functions
extend them. The default registrations live in :mod:`.default_builders`,
which imports these helpers and runs the register-calls at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .builder_protocols import AggregationBuilder
from .builder_protocols import AtomBuilder
from .builder_protocols import CircuitBuilder
from .builder_protocols import TransformationBuilder


if TYPE_CHECKING:
    from ...circuit.circuit import Circuit


default_atom_builders: dict[tuple[str, int, str], AtomBuilder] = dict()
default_circuit_builders: dict[str, Callable[[], Circuit]] = dict()
default_aggregation_builders: dict[str, AggregationBuilder] = dict()
default_transformation_builders: dict[tuple[str, str], TransformationBuilder] = dict()


def register_atom_builder(
    functor: str, arity: int, structure: str, builder: AtomBuilder
):
    """Register a default atom builder backed by the given predicate factory."""
    default_atom_builders[(functor, arity, structure)] = builder


def register_circuit_builder(structure: str, builder: CircuitBuilder):
    """Register a default circuit builder for the given structure."""
    default_circuit_builders[structure] = builder


def register_aggregation_builder(operation: str, builder: AggregationBuilder):
    """Register a default aggregation builder for the given operation."""
    default_aggregation_builders[operation] = builder


def register_transformation_builder(
    from_structure: str, to_structure: str, builder: TransformationBuilder
):
    """Register a default transformation builder for the given structure pair."""
    default_transformation_builders[(from_structure, to_structure)] = builder

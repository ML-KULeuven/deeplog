#  Copyright (c) 2024-2026. KU Leuven
"""DeepLog module implementations and utilities."""

from .aggregation_modules import AggregationModule
from .deeplog_module import DeepLogModule
from .deeplog_module import SupportsToModule
from .elementwise import ColumnwiseModule
from .elementwise import ElementwiseModule
from .module_circuit import ModuleCircuit
from .module_circuit import compose_modules
from .reshape import TransformationNotPossible
from .reshape import construct_transformation
from .reshape import reshape
from .reshape import simplify_module
from .sequential import Sequential
from .wrappers import WrappedModule


__all__ = [
    "AggregationModule",
    "compose_modules",
    "ColumnwiseModule",
    "DeepLogModule",
    "ElementwiseModule",
    "ModuleCircuit",
    "Sequential",
    "SupportsToModule",
    "TransformationNotPossible",
    "WrappedModule",
    "construct_transformation",
    "reshape",
    "simplify_module",
]

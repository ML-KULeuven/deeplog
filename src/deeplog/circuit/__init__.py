#  Copyright (c) 2024-2026. KU Leuven
"""Algebraic circuits: build one, knowledge-compile it, move it into another
algebra, and evaluate it as a torch module.
"""

from ..formula.ast import CircuitNode
from .circuit import Circuit
from .knowledge_compilation.dispatch import knowledge_compile
from .transform import transform_circuit


__all__ = [
    # Circuits
    "Circuit",
    "CircuitNode",
    "knowledge_compile",
    "transform_circuit",
]

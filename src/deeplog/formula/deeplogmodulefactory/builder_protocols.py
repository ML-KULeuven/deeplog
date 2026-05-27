#  Copyright (c) 2024-2026. KU Leuven
"""The builder protocols relevant for the DeepLogModuleFactory"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Protocol

import torch

from ...algebraic import HasStructure
from ...circuit import Circuit
from ...module import SupportsToModule
from ...shape import Shape
from ...symbol import Symbol


class InternalNode(SupportsToModule, HasStructure, Protocol):
    """An object that has a structure and can be turned into a DeepLogModule."""


AggregationBuilder = Callable[
    [
        InternalNode,  # Child node (raw, not yet compiled)
        list[Symbol],  # Variables
        Sequence[InternalNode],  # Params
        list[torch.Tensor],
    ],  # Domains
    InternalNode,
]
TransformationBuilder = Callable[[Shape], InternalNode]
AtomBuilder = Callable[[list[tuple]], InternalNode]
CircuitBuilder = Callable[[], Circuit]

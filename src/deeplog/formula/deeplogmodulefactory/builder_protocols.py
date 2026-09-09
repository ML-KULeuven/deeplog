#  Copyright (c) 2024-2026. KU Leuven
"""The builder protocols relevant for the DeepLogModuleFactory.

Builders are the *lowering* side: each turns a piece of the AST into a runtime
artifact, so they all return :class:`~deeplog.module.SupportsToModule` (something
the factory finalizes with ``.to_module()``) rather than a structure-bearing
node. The factory already tracks algebraic structure itself (via the registry
key, the leaf symbol's tag, or the transformation's target), so a builder output
has no reason to advertise it.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import torch

from ...circuit import Circuit
from ...module import SupportsToModule
from ...shape import Shape
from ...symbol import Symbol


#: Builds the module reducing a child over the joint domain of the binders.
AggregationBuilder = Callable[
    [
        SupportsToModule,  # Materialized child module (feeders composed in)
        list[Symbol],  # Variables
        Sequence[Any],  # Params (folded carriers; unused by sum)
        list[torch.Tensor],  # Domains
    ],
    SupportsToModule,
]
#: Builds the module converting a value of the given shape between one pair of
#: structures, chosen by that pair.
TransformationBuilder = Callable[[Shape], SupportsToModule]
#: Builds the module for one atom, from the ground argument tuples asked of it.
AtomBuilder = Callable[[list[tuple]], SupportsToModule]
CircuitBuilder = Callable[[], Circuit]

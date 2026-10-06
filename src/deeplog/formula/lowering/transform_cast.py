#  Copyright (c) 2024-2026. KU Leuven
"""Elementwise structure casts, the default transformation builders.

A cast is a structure conversion that has no operator/leaf analogue and
therefore cannot be expressed as a circuit transform, such as
``probability → logprobability`` via ``torch.log``. The lowering reads a cast's
source column into the module :func:`build_transform` returns.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

from ...module.deeplog_module import DeepLogModule
from ...shape import Shape
from ...shape import map_shape
from ..circuit_node import cast_name


# Elementwise functions that cast a tensor from one algebraic structure to
# another, keyed (from, to) as the default transformation builders are.
_CAST_FUNCTIONS: dict[tuple[str, str], Callable[[torch.Tensor], torch.Tensor]] = {
    ("boolean", "probability"): torch.clone,
    ("boolean", "real"): torch.clone,
    ("boolean", "logprobability"): torch.log,
    ("boolean", "mpe"): torch.clone,
    ("probability", "logprobability"): torch.log,
    ("logprobability", "probability"): torch.exp,
}


class _StructureCast(DeepLogModule):
    """Apply a prepared elementwise function to move values into a new algebra.

    The cast declares nothing about its own algebra: its *output symbols* are
    labelled with the target structure by :func:`build_transform`. A cast's
    inputs and outputs live in different algebras, so only per-symbol labels can
    record both.
    """

    def __init__(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        input_shape: Shape,
        output_shape: Shape,
    ):
        """Bind the elementwise function to the declared shapes."""
        super().__init__(input_shape, output_shape)
        self._func = func

    def forward(self, *inputs):
        """Apply the elementwise function to one or more tensors."""
        if len(inputs) == 1:
            return self._func(inputs[0])
        return tuple(self._func(x) for x in inputs)


def build_transform(
    input_shape: Shape, from_structure: str, to_structure: str
) -> _StructureCast:
    """Build the registered elementwise cast from ``from_structure`` to ``to_structure``.

    Looks up the cast in :data:`_CAST_FUNCTIONS`, names each output for the
    cast of its input (:func:`~deeplog.formula.circuit_node.cast_name`), and
    wraps both in a :class:`_StructureCast` module.
    """
    func = _CAST_FUNCTIONS[(from_structure, to_structure)]
    output_shape = map_shape(partial(cast_name, to_structure), input_shape)
    return _StructureCast(func, input_shape, output_shape)

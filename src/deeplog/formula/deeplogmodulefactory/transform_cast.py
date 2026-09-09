#  Copyright (c) 2024-2026. KU Leuven
"""Elementwise structure casts used by :class:`DeepLogModuleFactory`.

These are the *module-cast* leg of ``create_transformation``: a structure
conversion that has no operator/leaf analogue and therefore cannot be expressed
as a circuit transform (the canonical case being ``real → probability`` via
``torch.sigmoid``). The factory wraps the child module in the cast returned by
:func:`build_transform`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ...module import DeepLogModule
from ...shape import Shape
from ...shape import map_shape
from ...symbol import with_structure


# Elementwise functions that cast a tensor from one algebraic structure to
# another. Extend this table to support a new (from, to) pair; the keys are
# matched against the strings passed to register_transformation_builder.
_CAST_FUNCTIONS: dict[tuple[str, str], Callable[[torch.Tensor], torch.Tensor]] = {
    ("boolean", "probability"): lambda x: x,
    ("real", "probability"): torch.sigmoid,
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

    Looks up the cast in :data:`_CAST_FUNCTIONS`, re-tags the symbolic
    output shape with ``to_structure``, and wraps both in a
    :class:`_StructureCast` module.
    """
    func = _CAST_FUNCTIONS[(from_structure, to_structure)]
    output_shape = map_shape(
        lambda symbol: with_structure(
            ("transform", (to_structure,), symbol), to_structure
        ),
        input_shape,
    )
    return _StructureCast(func, input_shape, output_shape)

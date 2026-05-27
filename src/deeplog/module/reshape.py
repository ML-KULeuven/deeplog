"""
A module that implements transformations between different shapes.
"""

#  Copyright (c) 2024-2026. KU Leuven

import numpy as np
import torch

from ..shape import Shape
from ..shape import SymTensor
from ..util import as_tuple
from .deeplog_module import DeepLogModule
from .sequential import Sequential


class TransformationNotPossible(Exception):
    """
    This exception is raised when a requested transformation is not possible.
    """

    def __init__(
        self,
        in_shape: SymTensor | tuple[SymTensor, ...],
        out_shape: SymTensor | tuple[SymTensor, ...],
    ):
        """Record the incompatible input/output shapes that triggered the failure."""
        super().__init__(
            f"Cannot construct a transformation from {in_shape} to {out_shape}."
        )


def construct_transformation(input_shape: Shape, output_shape: Shape) -> DeepLogModule:
    """
    Construct a torch Module that transforms between the input and output shapes.
    """
    if isinstance(input_shape, SymTensor):
        return _IndexingTransform(input_shape, output_shape)
    return _TupleIndexingTransform(input_shape, output_shape)


def reshape(
    module: DeepLogModule,
    input: tuple[SymTensor, ...] | SymTensor | None = None,
    output: tuple[SymTensor, ...] | SymTensor | None = None,
) -> DeepLogModule:
    """Wrap ``module`` so its input and/or output shape matches the requested shapes.

    :param module: The module whose input and/or output should be reshaped.
    :param input: If provided, the new input shape for ``module``.
    :param output: If provided, the new output shape for ``module``.
    :return: A module that performs identically to the given module under the requested shapes.
    """
    if input is not None and input != module.get_input_shape():
        module = Sequential(
            construct_transformation(input, module.get_input_shape()), module
        )
    if output is not None and output != module.get_output_shape():
        module = Sequential(
            module, construct_transformation(module.get_output_shape(), output)
        )
    return module


def _gather_indices(labels: dict, output: SymTensor, *, raise_on: tuple) -> np.ndarray:
    """Build the flat gather indices for a single output SymTensor over ``labels``."""
    if output.array.size == 0:
        return np.array([], dtype=np.int64)
    try:
        return np.vectorize(lambda x: labels[x])(output.array).astype(np.int64)
    except KeyError as exc:
        raise TransformationNotPossible(*raise_on) from exc


class _IndexingTransform(DeepLogModule):
    """Flatten a SymTensor input and gather one or more output SymTensors from it."""

    _input_shape: SymTensor
    _output_shape: Shape

    def __init__(self, input_shape: SymTensor, output_shape: Shape):
        """Precompute one gather buffer per output entry over the flattened input."""
        super().__init__(input_shape, output_shape)
        labels = {label: index for index, label in enumerate(input_shape.array.flat)}
        outputs = as_tuple(output_shape)
        for i, out in enumerate(outputs):
            arr = _gather_indices(labels, out, raise_on=(input_shape, output_shape))
            self.register_buffer(f"_indices_{i}", torch.from_numpy(arr))
        self._n_outputs = len(outputs)
        self._is_tuple_output = isinstance(output_shape, tuple)
        self._input_ndim = input_shape.array.ndim

    def forward(self, input_tensor):
        """Flatten the input once, then gather each output's positions."""
        if self._input_ndim == 0:
            flat = input_tensor.unsqueeze(1)
        else:
            flat = torch.flatten(input_tensor, start_dim=1, end_dim=self._input_ndim)
        gathered = tuple(
            flat[:, getattr(self, f"_indices_{i}")] for i in range(self._n_outputs)
        )
        return gathered if self._is_tuple_output else gathered[0]


class _TupleIndexingTransform(DeepLogModule):
    """Select+flatten+concat a tuple input once, then gather one or more output SymTensors."""

    _input_shape: tuple[SymTensor, ...]
    _output_shape: Shape

    def __init__(self, input_shapes: tuple[SymTensor, ...], output_shape: Shape):
        """Precompute the shared input flatten/concat and a gather buffer per output."""
        super().__init__(input_shapes, output_shape)
        outputs = as_tuple(output_shape)

        # Union of input-tuple entries that any output reads from.
        # When an output is empty we still keep one entry around to preserve the batch dim.
        selected_set: set[int] = set()
        for out in outputs:
            if out.array.size == 0:
                if input_shapes:
                    selected_set.add(0)
                continue
            for i, shape in enumerate(input_shapes):
                if any(sym in out for sym in shape):
                    selected_set.add(i)
        selected = sorted(selected_set)
        relevant = [input_shapes[i] for i in selected]

        flat_symbols = (
            np.concatenate([s.array.flatten() for s in relevant])
            if relevant
            else np.array([])
        )
        labels = {label: index for index, label in enumerate(flat_symbols)}

        for i, out in enumerate(outputs):
            arr = _gather_indices(labels, out, raise_on=(input_shapes, output_shape))
            self.register_buffer(f"_indices_{i}", torch.from_numpy(arr))

        self._selected = tuple(selected)
        self._end_dims = tuple(s.array.ndim for s in relevant)
        self._n_outputs = len(outputs)
        self._is_tuple_output = isinstance(output_shape, tuple)

    def forward(self, *x):
        """Flatten+concat the selected inputs once, then gather each output's positions."""
        if not self._selected:
            empty = torch.zeros(1, 0)
            return (
                tuple(empty for _ in range(self._n_outputs))
                if self._is_tuple_output
                else empty
            )
        flat = [
            torch.flatten(x[i], start_dim=1, end_dim=end_dim)
            if end_dim > 0
            else x[i].unsqueeze(1)
            for i, end_dim in zip(self._selected, self._end_dims, strict=True)
        ]
        concat = flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)
        gathered = tuple(
            concat[:, getattr(self, f"_indices_{i}")] for i in range(self._n_outputs)
        )
        return gathered if self._is_tuple_output else gathered[0]


def simplify_module(module: DeepLogModule) -> DeepLogModule:
    """
    Return a DeepLogModule with a simplified public input signature.

    Operations:
    - Remove empty SymTensor inputs.
    - Deduplicate identical entries within each SymTensor.
    - Deduplicate identical SymTensor inputs (first occurrence kept).
    - Collapse single-element tuples to a single SymTensor.

    The returned module prepends a transformation (via construct_transformation)
    that reintroduces any removed arguments so the wrapped module receives its
    original input shape.
    """
    original_shape = module.get_input_shape()
    original_inputs = as_tuple(original_shape)

    simplified_inputs: list[SymTensor] = []
    for symtensor in original_inputs:
        if symtensor.is_empty():
            continue
        simplified_inputs.append(SymTensor(list(dict.fromkeys(symtensor))))

    simplified_inputs = list(dict.fromkeys(simplified_inputs))

    if len(simplified_inputs) == 0:
        simplified_shape: Shape = tuple()
    elif len(simplified_inputs) == 1:
        simplified_shape = simplified_inputs[0]
    else:
        simplified_shape = tuple(simplified_inputs)

    if simplified_shape == original_shape:
        return module

    transform = construct_transformation(simplified_shape, original_shape)
    return Sequential(transform, module)

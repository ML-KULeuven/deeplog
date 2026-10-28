#  Copyright (c) 2024-2026. KU Leuven
from collections.abc import Callable

import torch

from deeplog import DeepLogModule
from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog.symbol import Symbol


class DummyModule(DeepLogModule):
    def __init__(
        self,
        input_shape: SymTensor | tuple[SymTensor, ...],
        output_shape: SymTensor | tuple[SymTensor, ...],
        forward: Callable | None = None,
    ):
        super().__init__(input_shape, output_shape)
        self._forward_func = forward

    def forward(self, *args):
        if self._forward_func is None:
            return None
        return self._forward_func(*args)


class Counted(torch.nn.Module):
    """``function``, keeping the number of rows of each call."""

    def __init__(self, function: Callable):
        super().__init__()
        self.function = function
        self.rows: list[int] = []

    def forward(self, *inputs):
        self.rows.append(inputs[0].shape[0])
        return self.function(*inputs)


def values_along(variable: Symbol, values: torch.Tensor) -> WrappedModule:
    """``variable``'s ``values`` along its axis, behind its one column."""
    return WrappedModule(
        lambda empty: values.reshape(1, 1, -1).expand(empty.shape[0], 1, -1),
        SymTensor([]),
        SymTensor([variable]),
    )

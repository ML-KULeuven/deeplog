#  Copyright (c) 2024-2026. KU Leuven
from collections.abc import Callable

from deeplog import DeepLogModule
from deeplog import SymTensor


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

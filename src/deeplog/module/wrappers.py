#  Copyright (c) 2024-2026. KU Leuven
"""Utility wrapper modules.

- :class:`WrappedModule`: gives an arbitrary callable a ``DeepLogModule``
  shape contract.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..shape import Shape
from ..shape import get_all_symbols
from ..util import as_tuple
from .deeplog_module import DeepLogModule


def supply_empty_batch(module: DeepLogModule, args: tuple) -> tuple | None:
    """Pre-hook: synthesise a unit batch when a fully-baked module is called bare.

    When every declared input channel is zero-width — all features were baked
    into the wrapped module (a fully-constant circuit) — there is nothing to
    pass but the batch size, so a no-argument call stands in for "evaluate one
    row" and gets the empty ``(1, 0)`` tensor per channel that ``vmap`` maps
    over. Returning the new args replaces the call's inputs; this hook is
    *prepended*, so it runs before shape validation. Any non-bare call (or a
    real, non-zero-width input) is left untouched.

    Registered by every module that can end up outermost over a fully baked
    circuit — :class:`WrappedModule` and
    :class:`~deeplog.module.ColumnwiseModule` — since validation happens on the
    outermost module first.
    """
    if args:
        return None
    shapes = as_tuple(module.get_input_shape())
    if shapes and all(not list(get_all_symbols(shape)) for shape in shapes):
        return tuple(torch.zeros(1, 0) for _ in shapes)
    return None


class WrappedModule(DeepLogModule):
    """Wrap an arbitrary callable with DeepLogModule shape semantics."""

    def __init__(
        self,
        module: Callable,
        input_shape: Shape,
        output_shape: Shape,
        name: str | None = None,
        vmap: bool = False,
    ):
        """Wrap a callable with declared shapes and optional vmap execution.

        ``name`` defaults to the callable's own ``__name__`` when it has one (a
        function or a class), and to its type name otherwise — an ``nn.Module``
        instance is a callable without a ``__name__``, so wrapping one must not
        force the caller to invent a label.
        """
        super().__init__(input_shape, output_shape)
        self._module = module
        self.name = name or getattr(module, "__name__", type(module).__name__)
        self._vmap = vmap
        # Runs before the base class's shape-validation pre-hook so a fully-baked
        # (input-less) module can be called with no args; see _supply_empty_batch.
        self.register_forward_pre_hook(supply_empty_batch, prepend=True)

    def forward(self, *args, **kwargs):
        """Delegate execution to the wrapped callable."""
        if self._vmap:
            return torch.vmap(self._module)(*args, **kwargs)
        return self._module(*args, **kwargs)

    def __repr__(self):
        """Represent the wrapped callable including declared shapes."""
        return f"{self.name}({','.join(str(i) for i in as_tuple(self.get_input_shape()))})-> {self.get_output_shape()}"

#  Copyright (c) 2024-2026. KU Leuven
"""Utility wrapper modules.

- :class:`WrappedModule`: gives an arbitrary callable a ``DeepLogModule``
  shape contract.
- :class:`ConstantPrefillModule`: exposes a subset of an inner module's
  input slots as runtime inputs, prefilling the rest from a constant table.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..shape import Shape
from ..util import as_tuple
from .deeplog_module import DeepLogModule


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
        """Wrap a callable with declared shapes and optional vmap execution."""
        super().__init__(input_shape, output_shape)
        self._module = module
        self.name = name or module.__name__
        self._vmap = vmap

    def forward(self, *args, **kwargs):
        """Delegate execution to the wrapped callable."""
        if self._vmap:
            return torch.vmap(self._module)(*args, **kwargs)
        return self._module(*args, **kwargs)

    def __repr__(self):
        """Represent the wrapped callable including declared shapes."""
        return f"{self.name}({','.join(str(i) for i in as_tuple(self.get_input_shape()))})-> {self.get_output_shape()}"


class ConstantPrefillModule(nn.Module):
    """Expose a subset of an inner module's input slots as runtime inputs.

    The wrapped module expects ``n_total_slots`` inputs. The user-visible
    forward only takes a tensor whose last dim has size ``len(free_slot_indices)``
    — those values are scattered into the free slots, and the remaining
    slots are filled from ``constant_slot_values`` on every call.

    Useful whenever some inputs of an inner torch module are statically
    known constants (e.g. baked numeric labels, ablated features, pinned
    test inputs) and you don't want to rebuild the inner module to drop
    them.
    """

    # Declared so pyright sees the buffers as Tensors instead of
    # nn.Module's broad Tensor | Module union.
    _const_template: torch.Tensor
    _free_idx_buf: torch.Tensor

    def __init__(
        self,
        inner: nn.Module,
        n_total_slots: int,
        free_slot_indices: tuple[int, ...],
        constant_slot_values: dict[int, float],
    ):
        """Wrap ``inner`` so only ``free_slot_indices`` are user inputs."""
        super().__init__()
        self._inner = inner
        self._n_total = n_total_slots
        self._free = free_slot_indices
        # Stash the constant values as a dense tensor template so the
        # forward pass is a couple of indexed writes.
        const_template = torch.zeros(n_total_slots)
        for slot, value in constant_slot_values.items():
            const_template[slot] = value
        # Register as buffer so it follows .to(device) / .cuda()
        # without needing manual moves.
        self.register_buffer("_const_template", const_template)
        self._free_idx = torch.tensor(free_slot_indices, dtype=torch.long)
        self.register_buffer("_free_idx_buf", self._free_idx)

    def forward(self, x):
        """Scatter ``x`` into the free slots, prefill the rest with constants."""
        # x has shape (..., len(free_slots)) per the WrappedModule's
        # vmap convention. Build a (..., n_total) tensor by starting
        # from the constants and overwriting free slots. Out-of-place
        # ops only — vmap doesn't support in-place index_copy_.
        out_shape = x.shape[:-1] + (self._n_total,)
        full = self._const_template.to(dtype=x.dtype).expand(out_shape)
        idx = self._free_idx_buf.expand(out_shape[:-1] + (self._free_idx_buf.shape[0],))
        return self._inner(full.scatter(-1, idx, x))

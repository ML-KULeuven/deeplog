#  Copyright (c) 2024-2026. KU Leuven
"""Combine column groups of one module's output with a tensor operation.

:class:`ColumnwiseModule` applies a callable, typically an
:attr:`~deeplog.algebraic.AlgebraicStructure.operator_fns` entry, across groups
of one module's output columns, evaluating that module once.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..shape import SymTensor
from ..shape import get_all_symbols
from ..symbol import Symbol
from ..util import as_tuple
from .deeplog_module import DeepLogModule
from .wrappers import supply_empty_batch


class ColumnwiseModule(DeepLogModule):
    """Apply ``op`` across column groups of a *single* module's output.

    One module whose outputs are *all* the operands, the shape co-resident
    circuit roots are lowered into. ``inner`` is evaluated once per forward,
    where one module per operand would re-run the shared circuit once per
    operand.

    Groups are selected by symbol. The output keeps the *first* group's symbols
    by default — what a posterior wants; pass ``names`` to name them instead.
    """

    def __init__(
        self,
        op: Callable[..., Tensor],
        inner: DeepLogModule,
        *groups: tuple[Symbol, ...],
        name: str,
        names: tuple[Symbol, ...] | None = None,
    ) -> None:
        """Resolve each group's columns to indices into ``inner``'s output."""
        if not groups:
            raise ValueError("ColumnwiseModule needs at least one column group.")
        output_shape = inner.get_output_shape()
        if not isinstance(output_shape, SymTensor):
            raise ValueError(
                "ColumnwiseModule selects columns of a single SymTensor output, "
                f"got {output_shape!r}."
            )
        symbols = list(get_all_symbols(output_shape))
        for group in groups:
            missing = [symbol for symbol in group if symbol not in symbols]
            if missing:
                raise ValueError(
                    f"Column(s) {missing!r} are not among the module's outputs "
                    f"{symbols!r}."
                )
        if names is not None and len(names) != len(groups[0]):
            raise ValueError(
                f"ColumnwiseModule was given {len(names)} output name(s) for "
                f"{len(groups[0])} column(s)."
            )
        super().__init__(inner.get_input_shape(), SymTensor(list(names or groups[0])))
        self._op = op
        self._name = name
        self._inner = inner
        self._group_count = len(groups)
        for index, group in enumerate(groups):
            self.register_buffer(
                f"_group_{index}",
                torch.tensor([symbols.index(s) for s in group], dtype=torch.long),
                persistent=False,
            )
        # This is the outermost module over the compiled circuit, so a fully
        # baked (input-less) program has to stay callable bare through it.
        self.register_forward_pre_hook(supply_empty_batch, prepend=True)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Evaluate the inner module once, then combine its column groups."""
        y = as_tuple(self._inner(*x))[0]
        return self._op(
            *(
                y.index_select(-1, getattr(self, f"_group_{index}"))
                for index in range(self._group_count)
            )
        )

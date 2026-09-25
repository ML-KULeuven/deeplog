#  Copyright (c) 2024-2026. KU Leuven
"""Combine already-reduced module outputs with a tensor operation.

An operator reaches lowering as an AST node when its operands could not be
absorbed into a circuit — a connective over a quantifier, or any operator applied
to two separately lowered modules — so its operands are already numbers and it
becomes a tensor operation over them. An operator whose operands *are* nodes of
one circuit is a node of that circuit instead (:mod:`deeplog.circuit.split`).

:class:`ElementwiseModule` is not told *which* operation to apply: the callable
is an :attr:`~deeplog.algebraic.AlgebraicStructure.operator_fns` entry, so the
algebra supplies the semantics. Operands are reshaped to the union of their
inputs, which keeps the input tensors they declare, and combined column-wise.
They must agree on their output width or be a single column that broadcasts
across the others.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..shape import Shape
from ..shape import SymTensor
from ..shape import get_all_symbols
from ..shape import structures
from ..symbol import Symbol
from ..symbol import with_structure
from ..symbol import without_structure
from ..util import as_tuple
from .deeplog_module import DeepLogModule
from .reshape import reshape
from .wrappers import supply_empty_batch


def _output_columns(operand: DeepLogModule) -> list[Symbol]:
    """Return ``operand``'s output symbols, requiring a single tensor channel."""
    shape = operand.get_output_shape()
    if not isinstance(shape, SymTensor):
        raise ValueError(
            "ElementwiseModule combines single-tensor outputs, but an operand "
            f"declares {shape!r}."
        )
    return list(get_all_symbols(shape))


class ElementwiseModule(DeepLogModule):
    """Apply ``op`` to the outputs of ``operands`` over the union of their inputs."""

    def __init__(
        self,
        op: Callable[..., Tensor],
        *operands: DeepLogModule,
        name: str,
    ) -> None:
        """Reshape every operand to the union of the operands' inputs."""
        if not operands:
            raise ValueError("ElementwiseModule needs at least one operand.")
        structure = _shared_structure(operands)
        union = _union_input(operands)
        columns = [_output_columns(operand) for operand in operands]
        width = max(len(column) for column in columns)
        if any(len(column) not in (1, width) for column in columns):
            raise ValueError(
                f"ElementwiseModule operands must be {width} columns wide or a "
                f"single broadcasting column, got widths {[len(c) for c in columns]!r}."
            )
        super().__init__(
            union, SymTensor(_name_columns(name, columns, width, structure))
        )
        self._op = op
        self._operands = torch.nn.ModuleList(
            [reshape(operand, input=union) for operand in operands]
        )
        # This can be the outermost module over a compiled circuit, so a fully
        # baked (input-less) program has to stay callable bare through it.
        self.register_forward_pre_hook(supply_empty_batch, prepend=True)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Apply the operator to every operand's output, broadcasting single columns."""
        return self._op(*(as_tuple(operand(*x))[0] for operand in self._operands))


def _union_input(operands: tuple[DeepLogModule, ...]) -> Shape:
    """The operands' input tensors, in order, each symbol in the first that holds it.

    A symbol an earlier tensor holds is dropped from a later one, and a tensor
    left empty is dropped. A single tensor is the input itself.
    """
    placed: set[Symbol] = set()
    tensors: list[SymTensor] = []
    for operand in operands:
        for tensor in as_tuple(operand.get_input_shape()):
            fresh = [s for s in get_all_symbols(tensor) if s not in placed]
            placed.update(fresh)
            if fresh:
                tensors.append(SymTensor(fresh))
    if not tensors:
        return SymTensor([])
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def _shared_structure(operands: tuple[DeepLogModule, ...]) -> str | None:
    """Return the one algebraic structure every operand's outputs are labelled with.

    ``None`` when none of them are labelled: an operator over raw tensors is
    legitimate outside the formula layer. An operand whose *own* outputs mix
    structures is rejected like two operands that disagree.
    """
    found = {frozenset(structures(operand.get_output_shape())) for operand in operands}
    names = {name for group in found for name in group}
    if len(found) > 1 or len(names) > 1:
        raise ValueError(
            "ElementwiseModule operands must share a structure, got "
            f"{sorted(name or '<none>' for name in names)!r}."
        )
    return next(iter(names), None)


def _name_columns(
    name: str, columns: list[list[Symbol]], width: int, structure: str | None
) -> list[Symbol]:
    """Name each output column after the operator and the operand symbols it joins.

    Each minted name carries ``structure`` (``None`` only when the operands were
    unlabelled to begin with). The label goes on the *outside* and is stripped
    off the operand symbols it joins, giving ``divide(q1,z) _ probability``
    rather than the same tag nested once per operand.
    """
    minted = [
        (
            name,
            *(
                without_structure(column[i if len(column) > 1 else 0])
                for column in columns
            ),
        )
        for i in range(width)
    ]
    if structure is None:
        return minted
    return [with_structure(symbol, structure) for symbol in minted]


class ColumnwiseModule(DeepLogModule):
    """Apply ``op`` across column groups of a *single* module's output.

    The single-module dual of :class:`ElementwiseModule`: one module whose
    outputs are *all* the operands, the shape co-resident circuit roots are
    lowered into. ``inner`` is evaluated once per forward, where one module per
    operand would re-run the shared circuit once per operand.

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

#  Copyright (c) 2024-2026. KU Leuven
"""Predicate base classes and factories for DeepLog predicate modules."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping

import torch

from ...module.deeplog_module import DeepLogModule
from ...shape import SymTensor
from ...symbol import Symbol
from ...symbol import is_symbol


type Arguments = Iterable[tuple[Symbol, ...]]


class Predicate[*Ts](DeepLogModule, ABC):
    """Base class for predicates backed by torch computations or constants.

    Subclasses parameterize the generic with the tuple of tensor types that
    :meth:`forward_predicate` will receive — one entry per non-ignored
    argument position. Example: ``class MyPred(Predicate[torch.Tensor,
    torch.Tensor]):`` declares a binary predicate.
    """

    functor: str
    arity: int
    structure: str

    def __init__(
        self,
        all_arguments: Arguments,
        *,
        dedup: bool = False,
        ignore_argument: Iterable[int] | None = None,
    ):
        """Store predicate metadata and pre-process argument bindings/constants."""
        if dedup:
            all_arguments = list(dict.fromkeys(all_arguments))
        all_arguments = list(all_arguments)
        self._nr_evaluations = len(all_arguments)
        if not all(len(args) == self.arity for args in all_arguments):
            raise ValueError("Not all inputs are of the correct arity")

        ignore_argument = set(ignore_argument) if ignore_argument is not None else set()
        self._argument_indices = [
            i for i in range(self.arity) if i not in ignore_argument
        ]

        inputs, indices, constants = self._classify_arguments(all_arguments)

        input_shape = tuple(SymTensor(i) for i in inputs)
        evaluations: list[tuple] = [(self.functor, *args) for args in all_arguments]
        output_shape = SymTensor([("_", s, (self.structure,)) for s in evaluations])
        super().__init__(input_shape, output_shape)

        # Slots in ``_constants_{j}`` corresponding to symbol-valued arguments
        # are left uninitialized; ``_materialize_position`` must overwrite them
        # via ``base[:, indices] = x_j`` before the buffer is read.
        for k, j in enumerate(self._argument_indices):
            self.register_buffer(
                f"_indices_{j}",
                torch.tensor(indices[k], dtype=torch.long),
            )
            self.register_buffer(f"_constants_{j}", constants.get(j, torch.empty(0)))
        self._positions_with_constants: frozenset[int] = frozenset(constants)

    def _classify_arguments(
        self, all_arguments: list[tuple[Symbol, ...]]
    ) -> tuple[list[list[Symbol]], list[list[int]], dict[int, torch.Tensor]]:
        """Split arguments into symbol bindings and constant buffers.

        For each non-ignored position ``k`` in ``self._argument_indices``,
        ``inputs[k]`` collects the symbols and ``indices[k]`` records the
        evaluation row where each symbol appears. Constants are accumulated
        into ``constants[j]`` of shape ``[n_evaluations, *value.shape]``;
        slots for symbol-valued rows are left uninitialized.
        """
        inputs: list[list[Symbol]] = [[] for _ in self._argument_indices]
        indices: list[list[int]] = [[] for _ in self._argument_indices]
        constants: dict[int, torch.Tensor] = {}
        for i, arguments in enumerate(all_arguments):
            for k, j in enumerate(self._argument_indices):
                constant_or_symbol = self._resolve_argument(arguments[j], j)
                if is_symbol(constant_or_symbol):
                    inputs[k].append(constant_or_symbol)
                    indices[k].append(i)
                else:
                    value = torch.as_tensor(constant_or_symbol)
                    if j not in constants:
                        constants[j] = torch.empty(
                            self._nr_evaluations, *value.shape, dtype=value.dtype
                        )
                    constants[j][i] = value
        return inputs, indices, constants

    def _indices(self, j: int) -> torch.Tensor:
        return getattr(self, f"_indices_{j}")

    def _constants(self, j: int) -> torch.Tensor:
        return getattr(self, f"_constants_{j}")

    def _materialize_position(
        self, j: int, x_j: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor:
        """Build the ``[batch * n_evaluations, *feature_shape]`` input slice
        for argument position ``j``.

        ``x_j`` is the caller-supplied symbol input for this position, or
        ``None`` when no rows at this position are symbolic (eager case).
        """
        constants = self._constants(j)

        if x_j is None:
            base = constants.unsqueeze(0).expand(batch_size, *constants.shape)
            return base.reshape(-1, *constants.shape[1:])

        indices = self._indices(j).to(x_j.device)
        if j in self._positions_with_constants:
            # Constants are cast to x_j's dtype so the materialized input can
            # participate in autograd alongside the symbol values.
            base = constants.to(device=x_j.device, dtype=x_j.dtype)
            base = base.unsqueeze(0).expand(batch_size, *base.shape).contiguous()
        else:
            assert indices.numel() == self._nr_evaluations, (
                f"Position {j} has no constants but indices cover only "
                f"{indices.numel()}/{self._nr_evaluations} evaluations"
            )
            base = torch.empty(
                batch_size,
                self._nr_evaluations,
                *x_j.shape[2:],
                device=x_j.device,
                dtype=x_j.dtype,
            )
        base[:, indices] = x_j
        return base.reshape(-1, *base.shape[2:])

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Evaluate the predicate for all provided arguments and return batched results."""
        return self._run(list(x), batch_size=x[0].shape[0])

    def eager_eval(
        self, tensors: Mapping[Symbol, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Evaluate eagerly, resolving any free symbols via ``tensors``.

        Free symbols (positions where ``_resolve_argument`` returned a
        :class:`Symbol` rather than a constant) are looked up in
        ``tensors`` and stacked into the input batch for that position.
        Constant positions use the buffers populated during ``__init__``.

        Returns a tensor of shape ``[n_evaluations, *output_shape]``.
        Raises ``KeyError`` if a free symbol is missing from ``tensors``.
        """
        x_per_position = self._lookup_symbol_inputs(tensors or {})
        return self._run(x_per_position, batch_size=1).squeeze(0)

    def _lookup_symbol_inputs(
        self, tensors: Mapping[Symbol, torch.Tensor]
    ) -> list[torch.Tensor | None]:
        """Stack the tensors backing each position's free symbols.

        Returns one entry per position in ``self._argument_indices``: a
        ``[1, n_symbols, *feature_shape]`` tensor for positions that have
        free symbols, or ``None`` for fully-constant positions.

        Raises ``KeyError`` if a free symbol is missing from ``tensors``.
        """
        input_shapes = self.get_input_shape()
        x_per_position: list[torch.Tensor | None] = []
        for k, j in enumerate(self._argument_indices):
            sym_list = list(input_shapes[k])
            if not sym_list:
                x_per_position.append(None)
                continue
            looked_up = []
            for sym in sym_list:
                if sym not in tensors:
                    raise KeyError(
                        f"Predicate {self.functor}/{self.arity} has free "
                        f"symbol {sym} at position {j}; not present in "
                        f"the supplied tensors mapping."
                    )
                looked_up.append(tensors[sym])
            x_per_position.append(torch.stack(looked_up).unsqueeze(0))
        return x_per_position

    def _run(
        self,
        x_per_position: list[torch.Tensor | None],
        batch_size: int,
    ) -> torch.Tensor:
        """Materialize inputs, invoke ``forward_predicate``, and reshape.

        ``forward_predicate`` returns a flat tensor of shape
        ``[batch_size * n_evaluations, *output_feature_shape]``; the leading
        dim is split back into ``(batch_size, n_evaluations)`` here.
        """
        predicate_inputs = [
            self._materialize_position(j, x_per_position[k], batch_size)
            for k, j in enumerate(self._argument_indices)
        ]
        # `predicate_inputs` length is set by `len(_argument_indices)`, which
        # equals the subclass's declared arity — but pyright can't see that
        # the runtime list matches the static ``*Ts`` type-parameter tuple.
        y = self.forward_predicate(*predicate_inputs)  # pyright: ignore[reportArgumentType]
        return y.view(batch_size, self._nr_evaluations, *y.shape[1:])

    @classmethod
    def get_predicate(cls):
        """Return the tuple ``(functor, arity, structure)``."""
        return cls.functor, cls.arity, cls.structure

    def _resolve_argument(
        self, symbol: Symbol, index: int, /
    ) -> Symbol | int | float | bool | torch.Tensor:
        """
        Return a replacement for the argument at the given position.
        - Symbol: treated as a variable (the returned symbol is used instead of the original).
        - Non-symbol value: treated as a constant.
        """
        return symbol

    @abstractmethod
    def forward_predicate(self, *args: torch.Tensor) -> torch.Tensor:
        """Batched implementation of the predicate.

        Receives one tensor per non-ignored argument position, in argument
        index order, with constants and symbol values already substituted.
        Concrete subclasses can still document or narrow the individual
        tensor argument types they expect.
        """

    @classmethod
    def get_structure(cls) -> str:
        """Returns the name algebraic structure of the object"""
        return cls.structure

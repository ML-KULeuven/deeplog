#  Copyright (c) 2024-2026. KU Leuven
"""Predicate base classes and factories for DeepLog predicate modules."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence

import torch

from ...module.deeplog_module import DeepLogModule
from ...shape import SymTensor
from ...symbol import Symbol
from ...symbol import get_args
from ...symbol import get_predicate
from ...symbol import is_symbol
from ...symbol import is_variable
from ...symbol import symbol_to_pretty_string
from ...symbol import without_structure
from ...util import as_tuple
from ...variable import Domain
from ...variable import SymbolicDomain
from ...variable import ValueDomain


class Predicate[*Ts](DeepLogModule, ABC):
    """Base class for predicates backed by torch computations or constants.

    A predicate computes the atoms it is given, a column named by each, from
    their arguments. A compiler registers it under the predicate it computes.

    Subclasses parameterize the generic with the tuple of tensor types that
    :meth:`forward_predicate` will receive — one entry per non-ignored
    argument position. Example: ``class MyPred(Predicate[torch.Tensor,
    torch.Tensor]):`` declares a binary predicate.

    An argument position asks for each *distinct* symbol once, not once per
    evaluation: ``digit(i1,0) … digit(i1,9)`` are ten evaluations over one
    input, so the input shape holds ``i1`` once. Positions are expanded back
    to one row per evaluation before :meth:`forward_predicate` sees them,
    unless the position is listed in :attr:`distinct_arguments`.

    Each argument ranges over a domain, its sort (:meth:`domains_of`). A value
    written where it ranges over a named or a tensor domain is read as what a
    variable there holds: its position among the domain's names, or the number
    it writes. So an atom reads the same whether an aggregation feeds its
    variable a value or the value is written into it. Where it ranges over the
    values (:meth:`~deeplog.variable.Domain.of_values`), a written argument is
    what :meth:`resolve_argument` says: by default a number it writes, or else
    the name of an input.
    """

    #: Argument positions handed to :meth:`forward_predicate` as one row per
    #: distinct symbol rather than one row per evaluation. A predicate opts in
    #: when its computation is expensive and factors as "compute over this
    #: argument, then select using the others" — it must then map its result
    #: back to evaluations itself, using :meth:`evaluation_slots`. Positions
    #: carrying constants cannot be opted in.
    distinct_arguments: tuple[int, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass defining a name no predicate is read by any more.

        Raises:
            TypeError: If ``cls`` defines ``_resolve_argument``, the former name
                of :meth:`resolve_argument`, or declares ``functor``, ``arity``
                or ``structure``, which its key in a compiler states.
        """
        super().__init_subclass__(**kwargs)
        if "_resolve_argument" in vars(cls):
            raise TypeError(
                f"{cls.__qualname__} defines _resolve_argument, which is called "
                "resolve_argument now; rename the method."
            )
        declared = [
            name for name in ("functor", "arity", "structure") if name in vars(cls)
        ]
        if declared:
            raise TypeError(
                f"{cls.__qualname__} declares {', '.join(declared)}, which a "
                "predicate no longer does: a compiler registers it under its "
                "predicate, as atom_builders={(functor, arity, structure): "
                f"{cls.__qualname__}}}, and it names its columns by the atoms it "
                "is given."
            )

    def __init__(
        self,
        atoms: Iterable[Symbol],
        domains: Sequence[Domain] | Callable[[tuple[Symbol, ...]], Sequence[Domain]],
        *,
        ignore_arguments: Iterable[int] | None = None,
    ):
        """Compute ``atoms``, a column named by each, from their arguments.

        ``domains`` gives the domain of each argument, one per argument of the
        atoms, or is a function giving them for one atom's arguments
        (:meth:`~deeplog.formula.predicates.predicate.Predicate.domains_of`).

        Raises:
            ValueError: If ``atoms`` is empty or its atoms differ in arity,
                ``domains`` gives an atom other than one domain per argument, or
                a value written where an argument ranges over a named or a
                tensor domain is not one of its values.
        """
        atoms = list(atoms)
        self._domains = domains
        all_arguments = [get_args(without_structure(atom)) for atom in atoms]
        self._nr_evaluations = len(all_arguments)
        arities = {len(args) for args in all_arguments}
        if len(arities) != 1:
            raise ValueError(
                f"{type(self).__name__} computes atoms of one arity; it is given "
                f"atoms of arities {sorted(arities)}."
            )
        (arity,) = arities
        functor, _ = get_predicate(without_structure(atoms[0]))
        #: The predicate, as ``functor/arity``, that error messages name.
        self._name = f"{functor}/{arity}"

        ignore_arguments = (
            set(ignore_arguments) if ignore_arguments is not None else set()
        )
        self._argument_indices = [i for i in range(arity) if i not in ignore_arguments]

        inputs, indices, slots, constants = self._classify_arguments(all_arguments)

        input_shape = tuple(SymTensor(i) for i in inputs)
        super().__init__(input_shape, SymTensor(atoms))

        # Slots in ``_constants_{j}`` corresponding to symbol-valued arguments
        # are left uninitialized; ``_materialize_position`` must overwrite them
        # via ``base[:, indices] = ...`` before the buffer is read.
        for k, j in enumerate(self._argument_indices):
            self.register_buffer(
                f"_indices_{j}",
                torch.tensor(indices[k], dtype=torch.long),
            )
            self.register_buffer(
                f"_slots_{j}",
                torch.tensor(slots[k], dtype=torch.long),
            )
            self.register_buffer(f"_constants_{j}", constants.get(j, torch.empty(0)))
        self._positions_with_constants: frozenset[int] = frozenset(constants)

        overlap = sorted(set(self.distinct_arguments) & self._positions_with_constants)
        if overlap:
            raise ValueError(
                f"{type(self).__name__} lists argument position(s) {overlap} in "
                "distinct_arguments, but they carry constants; only fully "
                "symbolic positions can be passed unexpanded."
            )

    def _classify_arguments(
        self, all_arguments: list[tuple[Symbol, ...]]
    ) -> tuple[
        list[list[Symbol]], list[list[int]], list[list[int]], dict[int, torch.Tensor]
    ]:
        """Split arguments into symbol bindings and constant buffers.

        For each non-ignored position ``k`` in ``self._argument_indices``,
        ``inputs[k]`` collects the *distinct* symbols in order of first
        appearance, ``indices[k]`` records the evaluation row of each
        symbol-valued argument, and ``slots[k]`` records which entry of
        ``inputs[k]`` that row reads — the two together map the input tensor
        onto the evaluations. Constants are accumulated into ``constants[j]``
        of shape ``[n_evaluations, *value.shape]``; slots for symbol-valued
        rows are left uninitialized.
        """
        symbols: list[list[Symbol]] = [[] for _ in self._argument_indices]
        indices: list[list[int]] = [[] for _ in self._argument_indices]
        constants: dict[int, torch.Tensor] = {}
        for i, arguments in enumerate(all_arguments):
            resolved = self.resolve_arguments(arguments)
            for k, j in enumerate(self._argument_indices):
                constant_or_symbol = resolved[j]
                if is_symbol(constant_or_symbol):
                    symbols[k].append(constant_or_symbol)
                    indices[k].append(i)
                else:
                    value = torch.as_tensor(constant_or_symbol)
                    if j not in constants:
                        constants[j] = torch.empty(
                            self._nr_evaluations, *value.shape, dtype=value.dtype
                        )
                    constants[j][i] = value

        inputs: list[list[Symbol]] = []
        slots: list[list[int]] = []
        for per_evaluation in symbols:
            distinct = list(dict.fromkeys(per_evaluation))
            position = {symbol: s for s, symbol in enumerate(distinct)}
            inputs.append(distinct)
            slots.append([position[symbol] for symbol in per_evaluation])
        return inputs, indices, slots, constants

    def _indices(self, j: int) -> torch.Tensor:
        return getattr(self, f"_indices_{j}")

    def _slots(self, j: int) -> torch.Tensor:
        return getattr(self, f"_slots_{j}")

    def _constants(self, j: int) -> torch.Tensor:
        return getattr(self, f"_constants_{j}")

    def evaluation_slots(self, j: int) -> torch.Tensor:
        """Which input row of position ``j`` each evaluation reads.

        A ``[n_symbolic_evaluations]`` index into the distinct symbols of that
        position — the map a predicate listing ``j`` in
        :attr:`distinct_arguments` uses to expand its result back over the
        evaluations.
        """
        return self._slots(j)

    def _materialize_position(
        self, j: int, x_j: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor:
        """Build the ``[batch * n_evaluations, *feature_shape]`` input slice
        for argument position ``j``.

        ``x_j`` is the caller-supplied symbol input for this position — one
        row per distinct symbol — or ``None`` when no rows at this position
        are symbolic (eager case). Positions listed in
        :attr:`distinct_arguments` are handed over unexpanded instead, as
        ``[batch * n_distinct, *feature_shape]``.
        """
        constants = self._constants(j)

        if x_j is None:
            base = constants.unsqueeze(0).expand(batch_size, *constants.shape)
            return base.reshape(-1, *constants.shape[1:])

        if j in self.distinct_arguments:
            return x_j.reshape(-1, *x_j.shape[2:])

        # One row per symbol-valued evaluation, gathered from the distinct rows.
        expanded = x_j[:, self._slots(j).to(x_j.device)]
        if j in self._positions_with_constants:
            # Constants are cast to x_j's dtype so the materialized input can
            # participate in autograd alongside the symbol values.
            base = constants.to(device=x_j.device, dtype=x_j.dtype)
            base = base.unsqueeze(0).expand(batch_size, *base.shape).contiguous()
            base[:, self._indices(j).to(x_j.device)] = expanded
        else:
            assert self._indices(j).numel() == self._nr_evaluations, (
                f"Position {j} has no constants but indices cover only "
                f"{self._indices(j).numel()}/{self._nr_evaluations} evaluations"
            )
            base = expanded
        return base.reshape(-1, *base.shape[2:])

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Evaluate the predicate for all provided arguments and return batched results.

        A position whose input names no symbol holds only constants, and its
        tensor, which carries no values, is not read.
        """
        x_per_position = [
            None if symbols.is_empty() else x_j
            for symbols, x_j in zip(as_tuple(self.get_input_shape()), x, strict=True)
        ]
        return self._run(x_per_position, batch_size=x[0].shape[0])

    def eager_eval(
        self, tensors: Mapping[Symbol, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Evaluate eagerly, resolving any free symbols via ``tensors``.

        Free symbols (positions where ``resolve_argument`` returned a
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
                        f"{type(self).__name__} has free "
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
        # equals the atoms' arity — but pyright can't see that
        # the runtime list matches the static ``*Ts`` type-parameter tuple.
        y = self.forward_predicate(*predicate_inputs)  # pyright: ignore[reportArgumentType]
        return y.view(batch_size, self._nr_evaluations, *y.shape[1:])

    def _input_of(self, symbol: Symbol, index: int) -> tuple[int, int]:
        """Where ``symbol``, the argument at ``index`` of an atom, is read from.

        The index of the input tensor that argument position reads, and the
        column of ``symbol`` in it.

        Raises:
            ValueError: If no atom has ``symbol`` at ``index`` as an input.
        """
        if index not in self._argument_indices:
            raise ValueError(f"{self._name} ignores its argument {index + 1}.")
        position = self._argument_indices.index(index)
        symbols = list(as_tuple(self.get_input_shape())[position])
        if symbol not in symbols:
            raise ValueError(
                f"{self._name} reads no input {symbol_to_pretty_string(symbol)} "
                f"at its argument {index + 1}."
            )
        return position, symbols.index(symbol)

    def domains_of(self, arguments: tuple[Symbol, ...], /) -> tuple[Domain, ...]:
        """The domain each argument of an atom with ``arguments`` ranges over.

        Raises:
            ValueError: If the predicate gives other than one domain per
                argument.
        """
        found = tuple(
            self._domains(arguments) if callable(self._domains) else self._domains
        )
        if len(found) != len(arguments):
            raise ValueError(
                f"{self._name} gives {len(found)} domains for an atom of "
                f"{len(arguments)} arguments; it gives one per argument."
            )
        return found

    def resolve_arguments(
        self, arguments: tuple[Symbol, ...], /
    ) -> tuple[Symbol | int | float | bool | torch.Tensor, ...]:
        """What each argument of one atom stands for.

        A value written where an argument ranges over a named or a tensor
        domain (:meth:`domains_of`) is read as what a variable there holds; a
        variable, and an argument written where it ranges over the values, is
        what :meth:`resolve_argument` says. An ignored position keeps its
        symbol. A subclass whose reading of one argument depends on another in
        a way other than through its domain overrides this.

        Raises:
            ValueError: If a value written where an argument ranges over a named
                or a tensor domain is not one of its values.
        """
        domains = self.domains_of(arguments)
        resolved: list[Symbol | int | float | bool | torch.Tensor] = []
        for index, (symbol, domain) in enumerate(zip(arguments, domains, strict=True)):
            if index not in self._argument_indices:
                resolved.append(symbol)
                continue
            if is_variable(symbol) or isinstance(domain, ValueDomain):
                resolved.append(self.resolve_argument(symbol, index))
                continue
            value = value_of(symbol, domain)
            if value is None:
                raise ValueError(
                    f"{symbol_to_pretty_string(symbol)} is not a value of the "
                    f"domain of {self._name}'s argument {index + 1}."
                )
            resolved.append(value)
        return tuple(resolved)

    def resolve_argument(
        self, symbol: Symbol, index: int, /
    ) -> Symbol | int | float | bool | torch.Tensor:
        """What the argument ``symbol`` at position ``index`` of an atom stands for.

        It is asked for a variable, and for an argument written where it ranges
        over the values
        (:meth:`~deeplog.formula.predicates.predicate.Predicate.domains_of`). A
        subclass overrides this to give such arguments other constant values.
        By default a written number is that number, and any other argument is a
        variable, read from the input under its symbol.

        Returns:
            A symbol, which makes the argument a variable read from the input
            under that symbol, or a value, which makes it a constant: a tensor
            is used as it is, and a number or a boolean as a scalar.
        """
        if is_variable(symbol):
            return symbol
        number = number_of(symbol)
        return symbol if number is None else number

    @abstractmethod
    def forward_predicate(self, *args: torch.Tensor) -> torch.Tensor:
        """Batched implementation of the predicate.

        Receives one tensor per non-ignored argument position, in argument
        index order, with constants and symbol values already substituted.
        Concrete subclasses can still document or narrow the individual
        tensor argument types they expect.
        """


def value_of(symbol: Symbol, domain: Domain) -> int | float | None:
    """What a variable over ``domain`` holds where its value is ``symbol``.

    A named value is its position among the domain's names, and an unnamed one
    the number ``symbol`` writes, if the domain holds it. ``None`` if ``symbol``
    is no value of ``domain``.
    """
    if isinstance(domain, SymbolicDomain):
        return domain.index(symbol) if symbol in domain.names else None
    number = number_of(symbol)
    values = domain.as_tensor()
    if number is None or values.dim() != 1 or not bool((values == number).any()):
        return None
    return number


def number_of(symbol: Symbol) -> float | None:
    """The number ``symbol`` writes, or ``None`` if it writes none."""
    if len(symbol) != 1:
        return None
    try:
        return float(symbol[0])
    except ValueError:
        return None

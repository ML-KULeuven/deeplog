#  Copyright (c) 2024-2026. KU Leuven
"""Built-in predicate implementations."""

import functools
import math
from collections.abc import Callable
from collections.abc import Iterable

import torch

from ...symbol import Symbol
from ...symbol import is_variable
from ...symbol import symbol_to_pretty_string
from ...variable import Domain
from ...variable import SymbolicDomain
from .predicate import Predicate


class SumsPredicate(Predicate[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Predicate that checks whether a triplet of values x,y,z adhere to x+y=z."""

    functor = "sums"
    arity = 3
    structure = "boolean"

    def _resolve_argument(self, symbol: Symbol, _: int, /) -> float | Symbol:
        if len(symbol) != 1:
            return symbol
        try:
            return float(symbol[0])
        except (ValueError, TypeError, IndexError):
            return symbol

    def forward_predicate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return 1.0 where ``x + y == z`` and 0.0 otherwise."""
        return (x + y == z).to(torch.get_default_dtype())


class EqualityPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Predicate that checks the equality between pairs of symbols."""

    functor = "="
    arity = 2
    structure = "boolean"

    def __init__(
        self, all_arguments: Iterable[tuple[Symbol, Symbol]], domain: Iterable[Symbol]
    ):
        """
        Bind argument pairs to compare and map symbols in ``domain`` to integer ids so
        equality can be evaluated numerically.
        """
        self._domain_mapping = {symbol: i for i, symbol in enumerate(domain)}
        super().__init__(all_arguments)

    def _resolve_argument(self, symbol: Symbol, _: int, /) -> int | Symbol:
        return self._domain_mapping.get(symbol, symbol)

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Return a tensor of equality results between the two arguments."""
        return torch.eq(lhs, rhs).to(torch.get_default_dtype())


class _LabelProbabilityPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Shared logic for ``(atom, label)`` predicates with probability-space outputs.

    Subclasses set ``functor``/``structure`` and implement ``_convert_label``
    (literal → output-space scalar) and ``_negate`` (output-space ``1 - p``).
    """

    arity = 2

    def _resolve_argument(self, symbol: Symbol, index: int, /) -> float | Symbol:
        if index == 0:
            if symbol == ("true",):
                return 1.0
            if symbol == ("false",):
                return 0.0
            return symbol
        label_structure = "probability"
        if len(symbol) == 3 and symbol[0] == "_":
            label_structure = symbol[2][0]
            symbol = symbol[1]
        try:
            p = float(symbol[0])
        except ValueError:
            return "_", symbol, (label_structure,)
        return self._convert_label(p, label_structure)

    def _convert_label(self, p: float, label_structure: str) -> float:
        """Map a literal probability ``p`` (tagged ``label_structure``) into output space."""
        raise NotImplementedError

    def _negate(self, p: torch.Tensor) -> torch.Tensor:
        """Compute ``1 - p`` in the output space."""
        raise NotImplementedError

    def forward_predicate(self, atom: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Pick ``p`` where ``atom`` is true, else ``1 - p`` (in output space)."""
        return torch.where(atom.to(torch.bool), p, self._negate(p))


class ProbabilityPredicate(_LabelProbabilityPredicate):
    """``(atom, label)`` predicate with probability-space outputs."""

    functor = "p"
    structure = "probability"

    def _convert_label(self, p: float, label_structure: str) -> float:
        if label_structure == "logprobability":
            return math.exp(p)
        return p

    def _negate(self, p: torch.Tensor) -> torch.Tensor:
        return 1.0 - p


class LogProbabilityPredicate(_LabelProbabilityPredicate):
    """``(atom, label)`` predicate with log-probability-space outputs."""

    functor = "logp"
    structure = "logprobability"

    def _convert_label(self, p: float, label_structure: str) -> float:
        if label_structure == "probability":
            if p <= 0.0:
                return float("-inf")
            return math.log(p)
        return p

    def _negate(self, p: torch.Tensor) -> torch.Tensor:
        return torch.log1p(-torch.exp(p))


class NetworkPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Predicate that delegates evaluation to a provided ``torch.nn.Module``.

    Always binary: the first argument is fed to the module, and the second is a
    value that reads a row of its output. The ``arity`` constructor parameter is
    therefore expected to be ``2``.

    Given a ``domain``, which lists the values of the module's output rows in row
    order, an atom reads the row at its value's position in ``domain``: a written
    value resolves to an element of ``domain.as_tensor()``, and a variable is given
    one, as an aggregation over a binder ranging over ``domain`` does. Without a
    domain, a value is its row: a whole number from 0 up to the number of rows.

    Ground atoms sharing a first argument — ``digit(i1,0) … digit(i1,9)``, the
    usual "one classifier, N mutually exclusive values" pattern — are distinct
    evaluations over the *same* image. The first argument is therefore taken
    unexpanded (see :attr:`~deeplog.formula.predicates.predicate.Predicate.distinct_arguments`): the module runs once
    per distinct image, and :meth:`forward_predicate` selects each evaluation's
    row from that result.
    """

    distinct_arguments = (0,)

    #: The values of the module's output rows, in row order, or ``None`` when
    #: each value is its row.
    _values: torch.Tensor | None

    def __init__(
        self,
        functor: str,
        arity: int,
        structure: str,
        module: torch.nn.Module,
        all_arguments: Iterable[tuple[Symbol, ...]],
        domain: Domain | None = None,
    ):
        """Evaluate ``functor``/``arity`` atoms in ``structure`` through ``module``.

        Raises:
            ValueError: If ``domain``'s values are not distinct scalars, or a
                written value is not one of them, or, without a domain, not a
                row.
        """
        # functor/arity/structure/domain must exist before ``super().__init__``,
        # which resolves the arguments.
        self.functor = functor
        self.arity = arity
        self.structure = structure
        self._domain = domain
        values = None if domain is None else domain.as_tensor()
        if values is not None and (
            values.dim() != 1 or values.unique().numel() != values.numel()
        ):
            raise ValueError(
                f"The domain of {functor}/{arity} holds {tuple(values.shape)} values "
                "that are not distinct scalars, so they name no rows."
            )
        super().__init__(all_arguments)
        self._module = module
        self.register_buffer("_values", values, persistent=False)

    def _resolve_argument(self, symbol: Symbol, index: int, /) -> float | Symbol:
        value = _number(symbol)
        if index == 0:
            return symbol if value is None else value
        if is_variable(symbol):
            return symbol
        if isinstance(self._domain, SymbolicDomain):
            if symbol in self._domain.names:
                return self._domain.index(symbol)
        elif self._domain is not None:
            if value is not None and bool((self._domain.as_tensor() == value).any()):
                return value
        elif value is not None and value.is_integer() and value >= 0:
            return value
        raise ValueError(
            f"{symbol_to_pretty_string(symbol)} is not "
            + (
                f"a row of the module of {self.functor}/{self.arity}, which has no "
                "domain, so its values are its rows from 0."
                if self._domain is None
                else f"a value of the domain of {self.functor}/{self.arity}."
            )
        )

    def forward_predicate(
        self, inputs: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Run ``inputs`` through the wrapped module and read the row of each value.

        ``inputs`` holds one row per (batch item, *distinct* first argument);
        ``values`` one per (batch item, evaluation). The module therefore runs
        once per distinct image, and its output is expanded over the
        evaluations that share it before each evaluation reads the row of its
        value.

        Raises:
            ValueError: If the module's output does not have one row per value
                of the domain, or a value is not in the domain or, without a
                domain, not a row.
        """
        output = self._module(inputs)
        rows = self._rows(values, output)
        slots = self.evaluation_slots(0).to(output.device)
        batch_size = values.shape[0] // self._nr_evaluations
        output = output.view(batch_size, -1, *output.shape[1:])[:, slots]
        output = output.reshape(batch_size * self._nr_evaluations, *output.shape[2:])
        return output[torch.arange(output.shape[0]), rows]

    def _rows(self, values: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """The row of ``output`` each of ``values`` reads.

        Raises:
            ValueError: If ``output`` does not have one row per value of the
                domain, or a value is not in the domain or, without a domain,
                not a row.
        """
        if self._values is None:
            count = output.shape[1]
            found = (values == values.round()) & (values >= 0) & (values < count)
            if not bool(found.all()):
                raise ValueError(
                    f"{self.functor}/{self.arity} is given the value "
                    f"{values[~found][0].item()}, which is not a row of its "
                    f"module's {count} rows."
                )
            return values.to(torch.long)
        if output.dim() < 2 or output.shape[1] != len(self._values):
            raise ValueError(
                f"The module of {self.functor}/{self.arity} gives rows of shape "
                f"{tuple(output.shape[1:])}, where its domain has "
                f"{len(self._values)} values."
            )
        matches = torch.eq(values.unsqueeze(-1), self._values.to(values.device))
        found = matches.any(dim=-1)
        if not bool(found.all()):
            raise ValueError(
                f"{self.functor}/{self.arity} is given the value "
                f"{values[~found][0].item()}, which is not in its domain."
            )
        return matches.int().argmax(dim=-1)


def _number(symbol: Symbol) -> float | None:
    """The number ``symbol`` writes, or ``None`` if it writes none."""
    if len(symbol) != 1:
        return None
    try:
        return float(symbol[0])
    except ValueError:
        return None


def get_network_predicate(
    functor: str,
    arity: int,
    structure: str,
    module: torch.nn.Module,
    domain: Domain | None = None,
) -> Callable[[Iterable[tuple[Symbol, ...]]], NetworkPredicate]:
    """Curry :class:`NetworkPredicate` with ``(functor, arity, structure, module)`` and ``domain``.

    ``domain`` lists the values of ``module``'s output rows, in row order; without
    one, a value is its row. The returned callable accepts ``all_arguments`` and
    yields a configured predicate — suitable as an entry in an ``atom_builders``
    mapping.
    """
    return functools.partial(
        NetworkPredicate, functor, arity, structure, module, domain=domain
    )

#  Copyright (c) 2024-2026. KU Leuven
"""Built-in predicate implementations."""

import functools
import math
from collections.abc import Callable
from collections.abc import Iterable

import torch

from ...symbol import Symbol
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
        return (x + y == z).to(z.dtype)


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
        return torch.eq(lhs, rhs)


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
            if p < 1e-12:
                return float("-inf")
            return math.log(p)
        return p

    def _negate(self, p: torch.Tensor) -> torch.Tensor:
        return torch.log1p(-torch.exp(p))


class _NetworkPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Predicate that delegates evaluation to a provided ``torch.nn.Module``.

    Always binary: the first argument is fed to the module and the second
    selects a row from its output. The ``arity`` constructor parameter is
    therefore expected to be ``2``.
    """

    def __init__(
        self,
        functor: str,
        arity: int,
        structure: str,
        module: torch.nn.Module,
        all_arguments: Iterable[tuple[Symbol, ...]],
    ):
        # functor/arity/structure must exist before ``super().__init__`` —
        # the base reads ``self.arity`` during argument validation.
        self.functor = functor
        self.arity = arity
        self.structure = structure
        super().__init__(all_arguments)
        self._module = module

    def _resolve_argument(self, symbol: Symbol, _: int, /) -> float | Symbol:
        if len(symbol) == 1:
            try:
                return float(symbol[0])
            except ValueError:
                return symbol
        return symbol

    def forward_predicate(
        self, inputs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Run ``inputs`` through the wrapped module and gather rows by ``indices``."""
        output = self._module(inputs)
        return output[torch.arange(output.shape[0]), indices.to(torch.long)]


def get_network_predicate(
    functor: str, arity: int, structure: str, module: torch.nn.Module
) -> Callable[[Iterable[tuple[Symbol, ...]]], _NetworkPredicate]:
    """Curry :class:`_NetworkPredicate` with ``(functor, arity, structure, module)``.

    The returned callable accepts ``all_arguments`` and yields a configured
    predicate — suitable as an entry in an ``atom_builders`` mapping.
    """
    return functools.partial(_NetworkPredicate, functor, arity, structure, module)

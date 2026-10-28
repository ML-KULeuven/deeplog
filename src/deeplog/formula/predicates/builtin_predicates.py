#  Copyright (c) 2024-2026. KU Leuven
"""Built-in predicate implementations."""

import math
from collections.abc import Callable
from collections.abc import Iterable

import torch

from ...algebraic import BOOLEAN
from ...symbol import Symbol
from ...symbol import get_args
from ...symbol import get_predicate
from ...symbol import is_variable
from ...symbol import retag
from ...symbol import symbol_to_pretty_string
from ...symbol import without_structure
from ...variable import Domain
from .predicate import Predicate
from .predicate import number_of


#: The truth values, the domain of the atom a probability label weighs.
_TRUTH = Domain.of_structure(BOOLEAN)

#: The values, the domain of an argument read as it is.
_VALUES = Domain.of_values()


class EqualityPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Whether the two arguments of an atom name the same value.

    Both arguments range over the domain of the variable the atom compares, so
    a written value compared with a variable is read as the variable's value
    is. Two written values range over the names they write, so they compare by
    name.
    """

    def __init__(
        self,
        atoms: Iterable[Symbol],
        domain_of: Callable[[Symbol], Domain],
    ):
        """Compare the arguments of each atom, ``domain_of`` giving each variable's domain.

        Raises:
            ValueError: If a written value is not a value of the domain of the
                variable it is compared with, or two variables compared with
                each other range over different domains.
        """
        # Asked while the arguments are resolved: a compiled module keeps the
        # domains it read, and no compiler.
        self._domain_of = domain_of
        self._variables: dict[Symbol, Domain] = {}
        super().__init__(atoms, self._compared)
        del self._domain_of

    def _compared(self, arguments: tuple[Symbol, ...]) -> tuple[Domain, Domain]:
        """Both arguments' domain: the compared variable's, or the names written."""
        variable = next(
            (argument for argument in arguments if is_variable(argument)), None
        )
        if variable is None:
            names = Domain.of(dict.fromkeys(arguments))
            return names, names
        if variable not in self._variables:
            self._variables[variable] = self._domain_of(variable)
        return self._variables[variable], self._variables[variable]

    def resolve_arguments(
        self, arguments: tuple[Symbol, ...], /
    ) -> tuple[Symbol | int | float | bool | torch.Tensor, ...]:
        """The two arguments as variables, or as the values the variables are compared with."""
        lhs, rhs = arguments
        if is_variable(lhs) and is_variable(rhs):
            if self._compared((lhs,)) != self._compared((rhs,)):
                raise ValueError(
                    f"{symbol_to_pretty_string(lhs)} and "
                    f"{symbol_to_pretty_string(rhs)} range over different "
                    "domains, so their values cannot be compared."
                )
            return lhs, rhs
        return super().resolve_arguments(arguments)

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Return a tensor of equality results between the two arguments."""
        return torch.eq(lhs, rhs).to(torch.get_default_dtype())


class _LabelProbabilityPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Shared logic for ``(atom, label)`` predicates with probability-space outputs.

    The first argument is a truth value, a variable's or a written ``false`` or
    ``true``, and the second the probability that it is ``true``. Subclasses
    implement ``_convert_label`` (literal → output-space scalar) and
    ``_negate`` (output-space ``1 - p``).
    """

    def __init__(self, atoms: Iterable[Symbol]):
        """Weigh the truth value of each atom by its label."""
        super().__init__(atoms, (_TRUTH, _VALUES))

    def resolve_argument(self, symbol: Symbol, index: int, /) -> float | Symbol:
        if index == 0:
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

    def _convert_label(self, p: float, label_structure: str) -> float:
        if label_structure == "logprobability":
            return math.exp(p)
        return p

    def _negate(self, p: torch.Tensor) -> torch.Tensor:
        return 1.0 - p


class LogProbabilityPredicate(_LabelProbabilityPredicate):
    """``(atom, label)`` predicate with log-probability-space outputs."""

    def _convert_label(self, p: float, label_structure: str) -> float:
        if label_structure == "probability":
            if p <= 0.0:
                return float("-inf")
            return math.log(p)
        return p

    def _negate(self, p: torch.Tensor) -> torch.Tensor:
        return torch.log1p(-torch.exp(p))


class NetworkPredicate(Predicate[torch.Tensor]):
    """Predicate whose values are the output rows of a provided ``torch.nn.Module``.

    Always binary: the first argument is fed to the module, and the second is a
    value of ``domain``, which lists the values of the module's output rows in
    row order. ``digit(x, v)`` is the entry of ``x``'s row at ``v``'s position.

    It reads only the first arguments of the atoms it is given, and computes
    every value of each, running the module once per distinct first argument:
    asked for ``digit(i1, 2)`` and ``digit(i2, D)``, it computes ``digit(i1, 0)``
    to ``digit(i1, 9)`` and ``digit(i2, 0)`` to ``digit(i2, 9)``, in that order.
    A second argument only asks for an atom: a written value must be one of the
    domain's, written as the domain writes it
    (:attr:`~deeplog.variable.Domain.values`), and a variable may stand there.
    """

    def __init__(
        self,
        atoms: Iterable[Symbol],
        *,
        module: torch.nn.Module,
        domain: Domain,
    ):
        """Compute every value of each first argument of ``atoms`` through ``module``.

        Raises:
            ValueError: If ``domain`` does not write its values, an atom is not
                binary, its first argument is a written number, which names no
                input, or its second is a written value not in ``domain``.
        """
        atoms = list(atoms)
        functor, arity = (
            get_predicate(without_structure(atoms[0])) if atoms else ("network", 2)
        )
        name = f"{functor}/{arity}"
        try:
            values = domain.values
        except ValueError as error:
            raise ValueError(
                f"The domain of {name} does not write the values its rows stand "
                f"for: {error}"
            ) from None
        written = set(values)
        inputs: dict[Symbol, None] = {}
        for atom in atoms:
            arguments = get_args(without_structure(atom))
            if len(arguments) != 2:
                raise ValueError(f"{name} is binary; it is given {atom}.")
            first, second = arguments
            if number_of(first) is not None:
                raise ValueError(
                    f"{name} feeds its module the input its first argument names, "
                    f"and the number {symbol_to_pretty_string(first)} names none."
                )
            if not is_variable(second) and second not in written:
                raise ValueError(
                    f"{symbol_to_pretty_string(second)} is not a value of the "
                    f"domain of {name}."
                )
            inputs[first] = None
        computed = [
            retag((functor, first, value), atoms[0])
            for first in inputs
            for value in values
        ]
        super().__init__(computed, (_VALUES, domain), ignore_arguments=(1,))
        self._module = module
        self._rows = len(values)

    def _run(
        self, x_per_position: list[torch.Tensor | None], batch_size: int
    ) -> torch.Tensor:
        """The module's rows on the distinct first arguments, one entry per atom."""
        (inputs,) = x_per_position
        assert inputs is not None
        rows = self.forward_predicate(inputs.reshape(-1, *inputs.shape[2:]))
        return rows.reshape(batch_size, -1)

    def forward_predicate(self, inputs: torch.Tensor) -> torch.Tensor:
        """The module's output on ``inputs``, a row of one entry per value for each.

        Raises:
            ValueError: If the module's output does not have one row per input
                of one entry per value of the domain.
        """
        output = self._module(inputs)
        if output.shape[1:] != (self._rows,):
            raise ValueError(
                f"The module of {self._name} gives rows of shape "
                f"{tuple(output.shape[1:])}, where its domain has {self._rows} "
                "values."
            )
        return output

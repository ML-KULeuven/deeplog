#  Copyright (c) 2024-2026. KU Leuven
"""Built-in predicate implementations."""

import math
from collections.abc import Callable
from collections.abc import Iterable

import torch

from ...algebraic import BOOLEAN
from ...symbol import Symbol
from ...symbol import get_predicate
from ...symbol import is_variable
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


class NetworkPredicate(Predicate[torch.Tensor, torch.Tensor]):
    """Predicate that delegates evaluation to a provided ``torch.nn.Module``.

    Always binary: the first argument is fed to the module, and the second is a
    value that reads a row of its output.

    Given a ``domain``, which lists the values of the module's output rows in row
    order, the second argument ranges over it: an atom reads the row at its
    value's position, whether the value is written or a variable holds it.
    Without a domain, the second argument ranges over the values, and a written
    value is its row: a whole number from 0 up to the number of rows.

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
        atoms: Iterable[Symbol],
        *,
        module: torch.nn.Module,
        domain: Domain | None = None,
    ):
        """Compute ``atoms`` through ``module``.

        Raises:
            ValueError: If ``domain``'s values are not distinct scalars, or a
                written value is not one of them, or, without a domain, not a
                row.
        """
        atoms = list(atoms)
        values = None if domain is None else domain.as_tensor()
        if values is not None and (
            values.dim() != 1 or values.unique().numel() != values.numel()
        ):
            name = (
                "/".join(map(str, get_predicate(without_structure(atoms[0]))))
                if atoms
                else type(self).__name__
            )
            raise ValueError(
                f"The domain of {name} holds {tuple(values.shape)} values that "
                "are not distinct scalars, so they name no rows."
            )
        super().__init__(atoms, (_VALUES, _VALUES if domain is None else domain))
        self._module = module
        self.register_buffer("_values", values, persistent=False)

    def resolve_argument(self, symbol: Symbol, index: int, /) -> float | Symbol:
        """The module's input, or, without a domain, the row a written value reads.

        The first argument is a number it writes, or else a variable. Without a
        domain, the second is a variable or a row, a whole number from 0; with
        one, a written value is read as a value of the domain instead.

        Raises:
            ValueError: If, without a domain, the second argument writes no row.
        """
        value = number_of(symbol)
        if index == 0:
            return symbol if value is None else value
        if is_variable(symbol):
            return symbol
        if value is not None and value.is_integer() and value >= 0:
            return value
        raise ValueError(
            f"{symbol_to_pretty_string(symbol)} is not a row of the module of "
            f"{self._name}, which has no domain, so its values are its rows from 0."
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
                    f"{self._name} is given the value "
                    f"{values[~found][0].item()}, which is not a row of its "
                    f"module's {count} rows."
                )
            return values.to(torch.long)
        if output.dim() < 2 or output.shape[1] != len(self._values):
            raise ValueError(
                f"The module of {self._name} gives rows of shape "
                f"{tuple(output.shape[1:])}, where its domain has "
                f"{len(self._values)} values."
            )
        matches = torch.eq(values.unsqueeze(-1), self._values.to(values.device))
        found = matches.any(dim=-1)
        if not bool(found.all()):
            raise ValueError(
                f"{self._name} is given the value "
                f"{values[~found][0].item()}, which is not in its domain."
            )
        return matches.int().argmax(dim=-1)

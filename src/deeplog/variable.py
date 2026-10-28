#  Copyright (c) 2024-2026. KU Leuven
"""DeepLog variables and the domains they range over.

A DeepLog model declares its variables and, for each, the domain it ranges over.
Two consumers read that declaration: an aggregation enumerates its binder's
domain to substitute each value into the body, and knowledge compilation sizes
one compiler variable per declared domain.

A domain is *symbolic*, an ordered tuple of value names, whose position is the
value's identity; a *tensor* of values that have no names of their own, which
writes those that are numbers by their numerals; or the open domain of
*values*, such as numbers or images, which is never enumerated. Each answers
only the view it has: asking a tensor domain of images for its value names
raises rather than inventing them.

A variable is linked to the atoms it occurs in, not to atoms per value: see
:data:`VariableAtoms`.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor

from .algebraic import AlgebraicStructure
from .symbol import Symbol
from .symbol import apply_substitution
from .symbol import symbol_to_pretty_string
from .symbol import to_symbol


class Domain(ABC):
    """The values a variable ranges over, in value order."""

    @staticmethod
    def of(values: Iterable[Symbol | str | int]) -> Domain:
        """A domain of named values, in the given order."""
        return SymbolicDomain(
            tuple(
                to_symbol(value if isinstance(value, str | tuple) else str(value))
                for value in values
            )
        )

    @staticmethod
    def of_tensor(values: Tensor) -> Domain:
        """A domain whose values are the rows of ``values`` and have no names."""
        return TensorDomain(values)

    @staticmethod
    def of_values() -> Domain:
        """The open domain of values without names, such as numbers or images.

        It is never enumerated, so a variable over it is free: the caller gives
        it its value.
        """
        return ValueDomain()

    @staticmethod
    def of_structure(structure: AlgebraicStructure) -> Domain:
        """The domain a variable associated with ``structure`` inherits.

        Raises:
            ValueError: If ``structure`` declares no values, i.e. they are not
                finitely enumerable, so a variable over it needs a domain of its
                own.
        """
        if structure.values is None:
            raise ValueError(
                f"Structure '{structure.name}' declares no values, so a variable "
                f"over it inherits no domain; declare the variable's domain."
            )
        return SymbolicDomain(structure.values)

    @abstractmethod
    def __len__(self) -> int:
        """The number of values in the domain."""

    @abstractmethod
    def as_tensor(self) -> Tensor:
        """The values as an enumerable tensor, one row per value."""

    @property
    def values(self) -> tuple[Symbol, ...]:
        """The symbol writing each value, in value order.

        Raises:
            ValueError: If this domain's values have no names.
        """
        raise ValueError(
            f"{type(self).__name__} values have no names; it can only be "
            f"enumerated as a tensor."
        )

    def index(self, value: Symbol) -> int:
        """The position of the value ``value`` writes.

        Raises:
            ValueError: If this domain's values have no names, or ``value``
                writes none of them.
        """
        raise ValueError(
            f"{type(self).__name__} values have no names, so "
            f"{symbol_to_pretty_string(value)} writes none of them."
        )


@dataclass(frozen=True)
class SymbolicDomain(Domain):
    """A domain of named values, whose position in ``names`` is their identity.

    An enumeration binds a value by that position, which is the identity the
    atom builders resolve their arguments to.
    """

    names: tuple[Symbol, ...]

    def __len__(self) -> int:
        """The number of values in the domain."""
        return len(self.names)

    def as_tensor(self) -> Tensor:
        """The value positions, which is what an enumeration substitutes."""
        return torch.arange(len(self.names))

    @property
    def values(self) -> tuple[Symbol, ...]:
        """The value names, in value order."""
        return self.names

    def index(self, value: Symbol) -> int:
        """The position of ``value``.

        Raises:
            ValueError: If ``value`` is not in the domain.
        """
        try:
            return self.names.index(value)
        except ValueError:
            raise ValueError(f"{value} is not a value of this domain.") from None


@dataclass(frozen=True, eq=False)
class TensorDomain(Domain):
    """A domain whose values are the rows of a tensor and carry no names.

    Where the rows are finite real numbers, each is written by a numeral
    (:attr:`values`): an integer by its digits, and a float by the fewest
    digits that read back as it in the tensor's dtype, so 0.1 in float32 is
    written ``0.1``. Any other row, such as an image, has no writing.

    Two tensor domains are equal when they hold equal values in the same order.
    """

    tensor: Tensor

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is a tensor domain of equal values, in the same order."""
        if not isinstance(other, TensorDomain):
            return NotImplemented
        return self.tensor.shape == other.tensor.shape and bool(
            (self.tensor == other.tensor.to(self.tensor.device)).all()
        )

    def __hash__(self) -> int:
        """The hash of the values' shape, which equal domains share."""
        return hash(tuple(self.tensor.shape))

    def __len__(self) -> int:
        """The number of values in the domain."""
        return len(self.tensor)

    def as_tensor(self) -> Tensor:
        """The values themselves."""
        return self.tensor

    @property
    def values(self) -> tuple[Symbol, ...]:
        """The numeral writing each number, in value order.

        Raises:
            ValueError: If the values are not finite real numbers, which have
                no names, or the domain holds a number twice.
        """
        return tuple(self._positions)

    def index(self, value: Symbol) -> int:
        """The position of the number ``value`` writes, as :attr:`values` writes it.

        Raises:
            ValueError: If the values have no writing (:attr:`values`), or
                ``value`` is not one of their numerals.
        """
        position = self._positions.get(value)
        if position is None:
            raise ValueError(
                f"{symbol_to_pretty_string(value)} is not a value of this domain, "
                f"whose values are written {_listed(self.values)}."
            )
        return position

    @cached_property
    def _positions(self) -> dict[Symbol, int]:
        """The position of each number, by the numeral writing it.

        Raises:
            ValueError: If the values are not finite real numbers, or the domain
                holds a number twice.
        """
        numbers = self.tensor
        if (
            numbers.dim() != 1
            or numbers.is_complex()
            or numbers.dtype == torch.bool
            or not bool(torch.isfinite(numbers).all())
        ):
            raise ValueError(
                "TensorDomain values that are not finite real numbers have no "
                "names; it can only be enumerated as a tensor."
            )
        positions: dict[Symbol, int] = {}
        for position, number in enumerate(numbers.tolist()):
            numeral = (_numeral(number, numbers.dtype),)
            if numeral in positions:
                raise ValueError(
                    f"The tensor domain holds {numeral[0]} twice, so that numeral "
                    "names no single value."
                )
            positions[numeral] = position
        return positions


def _numeral(number: float, dtype: torch.dtype) -> str:
    """How a :class:`TensorDomain` of ``dtype`` writes ``number``."""
    if not dtype.is_floating_point:
        return str(number)
    for digits in range(1, 18):
        text = f"{number:.{digits}g}"
        if torch.tensor(float(text), dtype=dtype).item() == number:
            return repr(float(text))
    return repr(number)


def _listed(values: tuple[Symbol, ...]) -> str:
    """The first few of ``values``, as an error message shows them."""
    shown = ", ".join(symbol_to_pretty_string(value) for value in values[:5])
    return shown + (", ..." if len(values) > 5 else "")


@dataclass(frozen=True)
class ValueDomain(Domain):
    """The open domain of values without names, which is never enumerated."""

    def __len__(self) -> int:
        """Never returns.

        Raises:
            ValueError: Always, since the values are not enumerated.
        """
        raise ValueError(_NOT_ENUMERATED)

    def as_tensor(self) -> Tensor:
        """Never returns.

        Raises:
            ValueError: Always, since the values are not enumerated.
        """
        raise ValueError(_NOT_ENUMERATED)


_NOT_ENUMERATED = (
    "The domain of values is not enumerated: a variable over it is free, given "
    "its value by the caller, so no aggregation can bind it."
)


@dataclass(frozen=True)
class Variable:
    """A model variable and the domain it ranges over.

    A variable owns no atoms. An atom is a predicate applied to terms, and a
    term may be a variable, so a variable is *linked* to an atom by occurring in
    it -- see :data:`VariableAtoms`. Nothing pairs a value with an atom: the atom
    asserting a value is what substituting that value into an occurrence yields.
    """

    name: Symbol
    domain: Domain

    def __len__(self) -> int:
        """The number of values the variable ranges over."""
        return len(self.domain)


#: The term position an occurrence leaves open for the variable's value.
OPEN: Symbol = ("_",)

#: Where each variable occurs — the atoms it appears in, with its own term
#: position left :data:`OPEN`. ``digit(i1,_)`` is the occurrence of the variable
#: ``digit(i1)`` over ``0 ... 9``, and ``digit(i1,3)`` -- the atom asserting the
#: value 3 -- is what substituting yields.
#:
#: A variable may occur in several atoms and an atom may be an occurrence of
#: several variables, so neither direction is a function. In the degenerate case
#: the occurrence *is* :data:`OPEN`: the whole atom is the variable's position,
#: so the values are the atoms, which is what a disjunction over unrelated atoms
#: declares.
type VariableAtoms = Mapping[Variable, tuple[Symbol, ...]]


def atom_asserting(occurrence: Symbol, value: Symbol) -> Symbol:
    """The atom asserting ``value`` at ``occurrence``'s open position."""
    return apply_substitution(occurrence, {OPEN: value})


def indicated_values(
    variables: VariableAtoms,
) -> dict[Symbol, tuple[tuple[Variable, int], ...]]:
    """Which ``(variable, value position)`` each atom asserts, by substitution.

    The atom-keyed view, which is how a circuit pass meets a variable: it holds
    leaves, not variables. Every atom an occurrence can yield appears, whether or
    not the formula reaches it -- which values a circuit *uses* is the circuit's
    fact, not the model's.

    Raises:
        ValueError: If a variable's values have no names, so no atom can be
            written for them.
    """
    asserted: dict[Symbol, list[tuple[Variable, int]]] = {}
    for variable, occurrences in variables.items():
        for occurrence in occurrences:
            for position, value in enumerate(variable.domain.values):
                atom = atom_asserting(occurrence, value)
                asserted.setdefault(atom, []).append((variable, position))
    return {atom: tuple(values) for atom, values in asserted.items()}

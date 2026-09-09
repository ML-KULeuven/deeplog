#  Copyright (c) 2024-2026. KU Leuven
"""DeepLog variables and the domains they range over.

A DeepLog model declares its variables and, for each, the domain it ranges over.
Two consumers read that declaration: an aggregation enumerates its binder's
domain to substitute each value into the body, and knowledge compilation sizes
one compiler variable per declared domain.

A domain is either *symbolic* — an ordered tuple of value names, whose position
is the value's identity — or a *tensor* of values that have no names of their
own, such as images. Each answers only the view it has: asking a tensor domain
for its value names raises rather than inventing them.

A variable is linked to the atoms it occurs in, not to atoms per value: see
:data:`VariableAtoms`.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .algebraic import AlgebraicStructure
from .symbol import Symbol
from .symbol import apply_substitution
from .symbol import to_symbol


class Domain(ABC):
    """The values a variable ranges over, in value order."""

    @staticmethod
    def of(values: Iterable[Symbol | str | int]) -> SymbolicDomain:
        """A domain of named values, in the given order."""
        return SymbolicDomain(
            tuple(
                to_symbol(value if isinstance(value, str | tuple) else str(value))
                for value in values
            )
        )

    @staticmethod
    def of_tensor(values: Tensor) -> TensorDomain:
        """A domain whose values are the rows of ``values`` and have no names."""
        return TensorDomain(values)

    @staticmethod
    def of_structure(structure: AlgebraicStructure) -> SymbolicDomain:
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
        """The value names, in value order.

        Raises:
            ValueError: If this domain's values have no names.
        """
        raise ValueError(
            f"{type(self).__name__} values have no names; it can only be "
            f"enumerated as a tensor."
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
    """A domain whose values are the rows of a tensor and carry no names."""

    tensor: Tensor

    def __len__(self) -> int:
        """The number of values in the domain."""
        return len(self.tensor)

    def as_tensor(self) -> Tensor:
        """The values themselves."""
        return self.tensor


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

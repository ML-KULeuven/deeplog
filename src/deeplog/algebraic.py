#  Copyright (c) 2024-2026. KU Leuven
"""Contains code related to the Algebraic Structures."""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import Protocol
from typing import runtime_checkable

import torch
from torch import Tensor

from .symbol import Symbol


@runtime_checkable
class HasStructure(Protocol):
    """A protocol that specifies that this object has a structure."""

    def get_structure(self) -> str:
        """Returns the name algebraic structure of the object"""
        ...


# Type alias for operator functions
OperatorFn = Callable[..., Tensor]

# Type alias for constant resolution
ConstantFn = Callable[[Symbol], float | None]


def _default_constant_fn(symbol: Symbol) -> float | None:
    """Parse single-element numeric symbols as float constants."""
    if len(symbol) == 1:
        try:
            return float(symbol[0])
        except ValueError:
            return None
    return None


@dataclass
class AlgebraicStructure:
    """Defines an algebraic structure with operators."""

    name: str
    operator_fns: dict[str, OperatorFn] = field(default_factory=dict)
    constant_fn: ConstantFn = _default_constant_fn
    named_constants: dict[str, Symbol] = field(default_factory=dict)
    idempotent: bool = True

    def get_constant_value(self, symbol: Symbol) -> float | None:
        """Return the float value for a constant symbol, or None if not a constant."""
        return self.constant_fn(symbol)

    @cached_property
    def symbol_to_role(self) -> dict[Symbol, str]:
        """Reverse view of :attr:`named_constants`: symbol -> role name."""
        return {symbol: role for role, symbol in self.named_constants.items()}

    @property
    def operators(self) -> frozenset[str]:
        """Return the set of operator names for this structure."""
        return frozenset(self.operator_fns.keys())

    def get_operator_fn(self, operator: str) -> OperatorFn | None:
        """Return the torch function for the given operator, or None if not defined."""
        return self.operator_fns.get(operator)


class Semiring(AlgebraicStructure):
    """A semiring with product and sum operators."""

    def __init__(  # noqa: D107
        self,
        name: str,
        product: str = "times",
        product_fn: OperatorFn = lambda a, b: a * b,
        sum: str = "plus",
        sum_fn: OperatorFn = lambda a, b: a + b,
        zero: Symbol = ("0",),
        one: Symbol = ("1",),
        constant_fn: ConstantFn = _default_constant_fn,
        idempotent: bool = True,
    ):
        super().__init__(
            name=name,
            operator_fns={product: product_fn, sum: sum_fn},
            constant_fn=constant_fn,
            named_constants={"zero": zero, "one": one},
            idempotent=idempotent,
        )
        self.product = product
        self.sum = sum
        self.zero = zero
        self.one = one


class Algebra(Semiring):
    """A semiring extended with a negation operator."""

    def __init__(  # noqa: D107
        self,
        name: str,
        product: str = "times",
        product_fn: OperatorFn = lambda a, b: a * b,
        sum: str = "plus",
        sum_fn: OperatorFn = lambda a, b: a + b,
        zero: Symbol = ("0",),
        one: Symbol = ("1",),
        constant_fn: ConstantFn = _default_constant_fn,
        negation: str = "not",
        negation_fn: OperatorFn = lambda x: 1.0 - x,
        idempotent: bool = True,
    ):
        super().__init__(
            name=name,
            product=product,
            product_fn=product_fn,
            sum=sum,
            sum_fn=sum_fn,
            zero=zero,
            one=one,
            constant_fn=constant_fn,
            idempotent=idempotent,
        )
        self.negation = negation
        self.operator_fns[negation] = negation_fn


BOOLEAN = Algebra(
    name="boolean",
    product="and",
    product_fn=lambda a, b: torch.minimum(a, b),
    sum="or",
    sum_fn=lambda a, b: torch.maximum(a, b),
    negation="not",
    negation_fn=lambda x: 1.0 - x,
)

PROBABILITY = Algebra(
    name="probability",
    product="times",
    product_fn=lambda a, b: a * b,
    sum="plus",
    sum_fn=lambda a, b: a + b,
    negation="negate",
    negation_fn=lambda x: 1.0 - x,
    idempotent=False,
)

LOGPROBABILITY = Algebra(
    name="logprobability",
    product="times",
    product_fn=lambda a, b: a + b,
    sum="plus",
    sum_fn=lambda a, b: torch.logaddexp(a, b),
    negation="negate",
    negation_fn=lambda x: torch.log1p(-torch.exp(x)),
    idempotent=False,
)

structure_registry: dict[str, AlgebraicStructure] = {
    "boolean": BOOLEAN,
    "probability": PROBABILITY,
    "logprobability": LOGPROBABILITY,
}


def register_structure(structure: AlgebraicStructure) -> None:
    """Register a custom algebraic structure so it can be looked up by name.

    Once registered, the structure can be referenced by name in
    :class:`~deeplog.circuit.Circuit`, :func:`~deeplog.circuit.transform`,
    and anywhere else that accepts a structure name string.
    """
    structure_registry[structure.name] = structure


def get_algebraic_structure(name: str) -> AlgebraicStructure:
    """Look up an algebraic structure by name."""
    if name not in structure_registry:
        raise ValueError(
            f"Unknown structure '{name}'. Available: {list(structure_registry.keys())}"
        )
    return structure_registry[name]

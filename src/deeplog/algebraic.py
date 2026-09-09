#  Copyright (c) 2024-2026. KU Leuven
"""Contains code related to the Algebraic Structures."""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field

import torch
from torch import Tensor

from .symbol import FalseSymbol
from .symbol import Symbol
from .symbol import TrueSymbol


# Type alias for operator functions
OperatorFn = Callable[..., Tensor]

#: Resolves a symbol to the constant it names. ``None`` is not a failure — it says
#: the symbol names no constant of the structure, so it is an *atom* — an input of
#: the circuit it appears in. An implementation raises instead when a symbol names
#: a constant the structure does not have, which is a different answer from naming
#: none.
ConstantFn = Callable[[Symbol], float | None]


def _default_constant_fn(symbol: Symbol) -> float | None:
    """Parse single-element numeric symbols as float constants."""
    if len(symbol) == 1:
        try:
            return float(symbol[0])
        except ValueError:
            return None
    return None


def _boolean_constant_fn(symbol: Symbol) -> float | None:
    """Resolve the two boolean values; ``None`` for an atom.

    Raises:
        ValueError: If ``symbol`` is a number. Boolean has two values and they
            are named, so a numeral is neither a value nor an atom.
    """
    if symbol == TrueSymbol:
        return 1.0
    if symbol == FalseSymbol:
        return 0.0
    if _default_constant_fn(symbol) is not None:
        raise ValueError(
            f"{symbol} is not a value of 'boolean': its values are "
            f"{FalseSymbol} and {TrueSymbol}. Write the value's own name."
        )
    return None


def _times(a: Tensor, b: Tensor) -> Tensor:
    """Default semiring product: ordinary multiplication."""
    return a * b


def _plus(a: Tensor, b: Tensor) -> Tensor:
    """Default semiring sum: ordinary addition."""
    return a + b


def _complement(x: Tensor) -> Tensor:
    """Default complement: ``one - x``."""
    return 1.0 - x


def _divide(a: Tensor, b: Tensor) -> Tensor:
    """Default division, flooring the denominator at the smallest normal value.

    Derived per dtype, not pinned: a constant ``1e-12`` stores as exactly zero
    in float16, where the clamp would silently do nothing. Bounds the forward
    pass only — the gradient still overflows just above the floor.
    """
    return a / b.clamp_min(torch.finfo(b.dtype).tiny)


def _log_division(a: Tensor, b: Tensor) -> Tensor:
    """Log-space division: subtraction, flooring only ``-inf``.

    Do *not* mirror :func:`_divide`'s floor here: ``log(tiny)`` is ``-87.3`` in
    float32, and log space exists to hold probabilities well below that.
    ``finfo.min`` touches nothing but ``-inf``, so ``-inf - -inf`` is ``-inf``
    rather than ``nan``.
    """
    return a - b.clamp_min(torch.finfo(b.dtype).min)


@dataclass(kw_only=True, eq=False)
class AlgebraicStructure:
    """Defines an algebraic structure: a set of values with operators over them."""

    name: str
    #: The set of labels a formula over this structure takes (Def 1's ``A_R``),
    #: in value order, or ``None`` when they are not finitely enumerable. A
    #: variable associated with this structure inherits these as its domain
    #: (:meth:`~deeplog.variable.Domain.of_structure`).
    values: tuple[Symbol, ...] | None = None
    operator_fns: dict[str, OperatorFn] = field(default_factory=dict, repr=False)
    constant_fn: ConstantFn = field(default=_default_constant_fn, repr=False)

    def __post_init__(self) -> None:
        """Terminate the cooperative chain each subclass registers into."""

    @property
    def roles(self) -> dict[str, str]:
        """The operator name this structure spells each of its roles with.

        A role is an operator that an axiom names, so a bare structure declares
        none: its ``operator_fns`` are free-form. Each subclass adds the roles
        its axioms name, which is how one structure resolves another's
        operators — the same product whether it is spelled ``and`` or ``times``.
        """
        return {}

    @property
    def identities(self) -> dict[str, Symbol]:
        """The symbol this structure names each of its identity elements with.

        What :attr:`roles` is for operators, this is for the constants an axiom
        names: an identity crosses between structures by the role it plays, not
        by the value it happens to have.
        """
        return {}

    def get_constant_value(self, symbol: Symbol) -> float | None:
        """The value ``symbol`` names, or ``None`` when it names an atom.

        See :data:`ConstantFn` for the three answers a caller distinguishes.

        Raises:
            ValueError: If ``symbol`` names a constant this structure does not
                have.
        """
        return self.constant_fn(symbol)

    @property
    def operators(self) -> frozenset[str]:
        """Return the set of operator names for this structure."""
        return frozenset(self.operator_fns.keys())

    def get_operator_fn(self, operator: str) -> OperatorFn | None:
        """Return the torch function for the given operator, or None if not defined."""
        return self.operator_fns.get(operator)


@dataclass(kw_only=True, eq=False)
class Semiring(AlgebraicStructure):
    """A semiring: a product and a sum with their identities."""

    product: str = "times"
    product_fn: OperatorFn = field(default=_times, repr=False)
    sum: str = "plus"
    sum_fn: OperatorFn = field(default=_plus, repr=False)
    zero: Symbol = field(default=("0",), repr=False)
    one: Symbol = field(default=("1",), repr=False)

    def __post_init__(self) -> None:
        """Register the product and sum, and check the identities resolve.

        Raises:
            ValueError: If ``constant_fn`` does not resolve an identity's
                symbol, so nothing can evaluate it.
        """
        super().__post_init__()
        self.operator_fns[self.product] = self.product_fn
        self.operator_fns[self.sum] = self.sum_fn
        for name, symbol in (("zero", self.zero), ("one", self.one)):
            if self.constant_fn(symbol) is None:
                raise ValueError(
                    f"Structure '{self.name}' spells its '{name}' as {symbol}, "
                    f"which its constant_fn does not resolve to a value."
                )

    @property
    def roles(self) -> dict[str, str]:
        """Add the product and the sum."""
        return {**super().roles, "product": self.product, "sum": self.sum}

    @property
    def identities(self) -> dict[str, Symbol]:
        """Add the sum's and the product's identities."""
        return {**super().identities, "zero": self.zero, "one": self.one}


@dataclass(kw_only=True, eq=False)
class Algebra(Semiring):
    """A semiring with an involutive complement."""

    negation: str = "not"
    negation_fn: OperatorFn = field(default=_complement, repr=False)

    def __post_init__(self) -> None:
        """Register the complement."""
        super().__post_init__()
        self.operator_fns[self.negation] = self.negation_fn

    @property
    def roles(self) -> dict[str, str]:
        """Add the complement."""
        return {**super().roles, "negation": self.negation}


@dataclass(kw_only=True, eq=False)
class Semifield(Semiring):
    """A semiring whose product is invertible, so division is defined.

    Independent of :class:`Algebra`: an invertible product says nothing about a
    complement. An algebra that divides differently supplies its own
    ``division_fn``.
    """

    division: str = "divide"
    division_fn: OperatorFn = field(default=_divide, repr=False)

    def __post_init__(self) -> None:
        """Register division."""
        super().__post_init__()
        self.operator_fns[self.division] = self.division_fn

    @property
    def roles(self) -> dict[str, str]:
        """Add the division."""
        return {**super().roles, "division": self.division}


@dataclass(kw_only=True, eq=False)
class _ComplementedSemifield(Algebra, Semifield):
    """The combination :data:`PROBABILITY` and :data:`LOGPROBABILITY` share.

    Private: it names no new axiom, so it is not a rung of the taxonomy — it
    exists because Python reifies an axiom set as a type.
    """


#: The two truth values under ``and`` / ``or`` / ``not``, with ``false`` as zero
#: and ``true`` as one. The structure a formula is written in before any reading
#: is put on it.
BOOLEAN = Algebra(
    name="boolean",
    values=(FalseSymbol, TrueSymbol),
    constant_fn=_boolean_constant_fn,
    zero=FalseSymbol,
    one=TrueSymbol,
    product="and",
    product_fn=lambda a, b: torch.minimum(a, b),
    sum="or",
    sum_fn=lambda a, b: torch.maximum(a, b),
    negation="not",
    negation_fn=_complement,
)

#: Probabilities in ``[0, 1]`` under ``times`` and ``plus``, with ``negate`` the
#: complement ``1 - p``. Its product is invertible, which is what a conditional
#: needs to divide by its evidence.
PROBABILITY = _ComplementedSemifield(
    name="probability",
    product="times",
    product_fn=_times,
    sum="plus",
    sum_fn=_plus,
    negation="negate",
    negation_fn=_complement,
)

#: :data:`PROBABILITY` in log space, where a product adds and a sum is
#: ``logaddexp``. Zero is ``-inf`` and one is ``0``.
LOGPROBABILITY = _ComplementedSemifield(
    name="logprobability",
    zero=("-inf",),
    one=("0",),
    product="times",
    product_fn=lambda a, b: a + b,
    sum="plus",
    sum_fn=lambda a, b: torch.logaddexp(a, b),
    negation="negate",
    negation_fn=lambda x: torch.log1p(-torch.exp(x)),
    division_fn=_log_division,
)


#: Max-product — the most probable explanation. A semiring like any other, so a
#: knowledge-compiled formula reaches it through the ordinary transform rather
#: than through a compile-time override — ``max`` in place of a sum keeps the
#: single best explanation where ``plus`` would total them all.
MPE = _ComplementedSemifield(
    name="mpe",
    product="times",
    product_fn=_times,
    sum="plus",
    sum_fn=torch.maximum,
    negation="negate",
    negation_fn=_complement,
)

#: Raw real-valued tensors — a network's pre-activation output. It defines no
#: operators, so nothing can be computed *in* it and no circuit can be built
#: over it; it exists so that "unlabelled" and "real-valued" stay distinct.
#: A module feeding the ``real -> probability`` sigmoid cast labels its outputs
#: with this rather than leaving them bare, because a missing label must never
#: select a conversion.
REAL = AlgebraicStructure(name="real")

structure_registry: dict[str, AlgebraicStructure] = {
    "boolean": BOOLEAN,
    "probability": PROBABILITY,
    "logprobability": LOGPROBABILITY,
    "mpe": MPE,
    "real": REAL,
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


def structure_name(structure: str | AlgebraicStructure) -> str:
    """Return the registry name for a structure given as a name *or* an object.

    Used at the string boundary (registry keys, leaf tags) where a name is
    required. Safe for custom structures passed by object: it
    reads ``.name`` directly and never resolves through ``structure_registry``,
    so a structure that was never globally registered still works.
    """
    return structure if isinstance(structure, str) else structure.name

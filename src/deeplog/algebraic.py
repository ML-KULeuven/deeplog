#  Copyright (c) 2024-2026. KU Leuven
"""Contains code related to the Algebraic Structures."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import SupportsIndex

import torch
from torch import Tensor

from .symbol import FalseSymbol
from .symbol import Symbol
from .symbol import TrueSymbol


# Type alias for operator functions
OperatorFn = Callable[..., Tensor]

#: Aggregates a body over the assignments of its binders. It receives the body's
#: values and then each param's, each stacked over the assignments into a tensor
#: of shape ``(batch, assignments, 1)``, and returns the aggregation's value, of
#: shape ``(batch, 1)``.
AggregationFn = Callable[..., Tensor]

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


def _unit_complement(x: Tensor) -> Tensor:
    """The complement in ``[0, 1]``: ``1 - x``, and zero where ``x`` exceeds one."""
    return (1.0 - x).clamp_min(0)


def _log_complement(x: Tensor) -> Tensor:
    """The complement of a probability in log space, ``log(1 - exp(x))``.

    It is ``-inf``, with a zero gradient, where ``x`` is at least zero. Below,
    it is ``log(-expm1(x))`` above ``-ln 2`` and ``log1p(-exp(x))`` under it,
    each where it is accurate.
    """
    inside = x < 0
    near = inside & (x > -math.log(2))
    far = inside & ~near
    placeholder = torch.full_like(x, -1.0)
    close = torch.log(-torch.expm1(torch.where(near, x, placeholder)))
    distant = torch.log1p(-torch.exp(torch.where(far, x, placeholder)))
    return torch.where(
        near, close, torch.where(far, distant, torch.full_like(x, -math.inf))
    )


def _log_add(a: Tensor, b: Tensor) -> Tensor:
    """``logaddexp``, and ``-inf`` with a zero gradient where both are ``-inf``."""
    both = (a == -math.inf) & (b == -math.inf)
    added = torch.logaddexp(torch.where(both, 0.0, a), torch.where(both, 0.0, b))
    return torch.where(both, torch.full_like(added, -math.inf), added)


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


def _sum_over(values: Tensor) -> Tensor:
    """``values`` added over the assignments."""
    return values.sum(dim=1)


def _log_sum_over(values: Tensor) -> Tensor:
    """``values``, logarithms, added over the assignments in log space.

    It is ``-inf``, with a zero gradient, where every value is.
    """
    empty = (values == -math.inf).all(dim=1, keepdim=True)
    summed = torch.logsumexp(torch.where(empty, 0.0, values), dim=1)
    return torch.where(empty.squeeze(1), torch.full_like(summed, -math.inf), summed)


def _max_over(values: Tensor) -> Tensor:
    """The greatest of ``values`` over the assignments."""
    return values.amax(dim=1)


def _min_over(values: Tensor) -> Tensor:
    """The least of ``values`` over the assignments."""
    return values.amin(dim=1)


@dataclass(kw_only=True, eq=False)
class AlgebraicStructure:
    """Defines an algebraic structure: a set of values, with operators and aggregators over them."""

    name: str
    #: The set of labels a formula over this structure takes (Def 1's ``A_R``),
    #: in value order, or ``None`` when they are not finitely enumerable. A
    #: variable associated with this structure inherits these as its domain
    #: (:meth:`~deeplog.variable.Domain.of_structure`).
    values: tuple[Symbol, ...] | None = None
    operator_fns: dict[str, OperatorFn] = field(default_factory=dict, repr=False)
    #: The aggregators a formula over this structure can aggregate with (Def 1's
    #: ``Agg_R``), by the operation an aggregation names.
    aggregation_fns: dict[str, AggregationFn] = field(default_factory=dict, repr=False)
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

    @property
    def aggregations(self) -> frozenset[str]:
        """The operations this structure has an aggregator for."""
        return frozenset(self.aggregation_fns)

    def get_aggregation_fn(self, operation: str) -> AggregationFn | None:
        """The aggregator for ``operation``, or ``None`` if the structure has none."""
        return self.aggregation_fns.get(operation)

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple:
        """Pickle a registered structure by its name, so it unpickles to itself."""
        if structure_registry.get(self.name) is self:
            return get_algebraic_structure, (self.name,)
        return super().__reduce_ex__(protocol)


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
    """The combination :data:`PROBABILITY`, :data:`LOGPROBABILITY` and :data:`MPE` share.

    Private: it names no new axiom, so it is not a rung of the taxonomy — it
    exists because Python reifies an axiom set as a type.
    """


#: The two truth values under ``and`` / ``or`` / ``not``, with ``false`` as zero
#: and ``true`` as one, aggregated by ``exists`` and ``forall``. The structure a
#: formula is written in before any reading is put on it.
BOOLEAN = Algebra(
    name="boolean",
    values=(FalseSymbol, TrueSymbol),
    constant_fn=_boolean_constant_fn,
    zero=FalseSymbol,
    one=TrueSymbol,
    product="and",
    product_fn=torch.minimum,
    sum="or",
    sum_fn=torch.maximum,
    negation="not",
    negation_fn=_complement,
    aggregation_fns={"exists": _max_over, "forall": _min_over},
)

#: Probabilities in ``[0, 1]`` under ``times`` and ``plus``, with ``negate`` the
#: complement ``1 - p``, aggregated by ``sum``. Its product is invertible, which is
#: what a conditional needs to divide by its evidence.
PROBABILITY = _ComplementedSemifield(
    name="probability",
    product="times",
    product_fn=_times,
    sum="plus",
    sum_fn=_plus,
    negation="negate",
    negation_fn=_unit_complement,
    aggregation_fns={"sum": _sum_over},
)

#: :data:`PROBABILITY` in log space, where a product adds and a sum is
#: ``logaddexp``, so ``sum`` aggregates by log-sum-exp. Zero is ``-inf`` and one
#: is ``0``.
LOGPROBABILITY = _ComplementedSemifield(
    name="logprobability",
    zero=("-inf",),
    one=("0",),
    product="times",
    product_fn=_plus,
    sum="plus",
    sum_fn=_log_add,
    negation="negate",
    negation_fn=_log_complement,
    division_fn=_log_division,
    aggregation_fns={"sum": _log_sum_over},
)


#: Max-product — the most probable explanation. A semiring like any other, so a
#: knowledge-compiled formula reaches it through the ordinary transform rather
#: than through a compile-time override — ``max`` in place of a sum keeps the
#: single best explanation where ``plus`` would total them all, and ``sum``
#: aggregates by it too.
MPE = _ComplementedSemifield(
    name="mpe",
    product="times",
    product_fn=_times,
    sum="plus",
    sum_fn=torch.maximum,
    negation="negate",
    negation_fn=_unit_complement,
    aggregation_fns={"sum": _max_over},
)

#: The real numbers under ``times`` and ``plus``, aggregated by ``sum``. A
#: network's pre-activation output is real, and so is a count, since the
#: ``boolean -> real`` cast reads ``true`` as 1 and ``false`` as 0. Its values
#: are not finitely enumerable.
REAL = Semiring(name="real", aggregation_fns={"sum": _sum_over})

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
    :class:`~deeplog.circuit.Circuit`, :func:`~deeplog.circuit.transform_circuit`,
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

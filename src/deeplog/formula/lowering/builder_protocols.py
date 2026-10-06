#  Copyright (c) 2024-2026. KU Leuven
"""The builder protocols a :class:`~deeplog.formula.lowering.compiler.Compiler` is configured with.

Builders are the *lowering* side: each turns a piece of the AST into a module.
An atom or cast builder returns the module for the atoms or values it is given,
and an aggregation builder a module for each aggregation, since several may
share one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

from ...module.deeplog_module import DeepLogModule
from ...shape import Shape
from ...symbol import Symbol


if TYPE_CHECKING:
    from ...variable import Domain
    from ..ast import Aggregation
    from .lowering import Lowering


class AggregationBuilder(Protocol):
    """Builds the modules computing the aggregations of one operation.

    Any callable with this signature is a builder.
    """

    def __call__(
        self, nodes: Sequence[Aggregation], lowering: Lowering, /
    ) -> Sequence[tuple[DeepLogModule, int]]:
        """For each node, in order, the module computing its value and the column holding it.

        ``nodes`` are the aggregations of the builder's operation in a level,
        unlowered, apart from any that binds a variable another reads free or
        binds a name another binds over another domain; ``lowering.compiler``
        declares the domain of each of their binders. The builder lowers what
        beneath them its method needs through ``lowering``. A column is a position among the module's output symbols,
        and the symbol there must be labelled with the node's algebra,
        :attr:`~deeplog.formula.ast.Aggregation.structure`. Several
        nodes may share a module, which the compiled module evaluates once. The
        lowering names each reported column, so the module's own names do not
        matter.
        """
        ...


class AtomBuilder(Protocol):
    """Builds the module computing the atoms of one predicate.

    A compiler registers a builder under the predicate it builds. Any callable
    with this signature is a builder; so is every
    :class:`~deeplog.formula.predicates.predicate.Predicate` subclass.
    """

    def __call__(self, atoms: Sequence[Symbol], /) -> DeepLogModule:
        """The module with a column named by each of ``atoms``.

        ``atoms`` are every atom of the builder's predicate that a level asks
        for, each labelled with its algebra. The module declares the domain of
        each argument of its atoms (:class:`DeclaresDomains`).
        """
        ...


@runtime_checkable
class DeclaresDomains(Protocol):
    """A module that declares the domain each argument of its atoms ranges over.

    Every :class:`~deeplog.formula.predicates.predicate.Predicate` is one, and
    every module an atom builder returns is: the compiler gives each variable
    the domain of the arguments it is
    (:class:`~deeplog.formula.lowering.sorts.Sorts`).
    """

    def domains_of(self, arguments: tuple[Symbol, ...], /) -> tuple[Domain, ...]:
        """The domain each argument of an atom with ``arguments`` ranges over."""
        ...


class TransformationBuilder(Protocol):
    """Builds the module casting values between the pair of algebras it is keyed by.

    Any callable with this signature is a builder.
    """

    def __call__(self, shape: Shape, /) -> DeepLogModule:
        """The module taking values named by ``shape`` into the target algebra."""
        ...

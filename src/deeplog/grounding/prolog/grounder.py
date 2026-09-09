#  Copyright (c) 2024-2026. KU Leuven
"""The abstract plain-Prolog grounder interface.

A grounder turns a logic program + goal into a proof formula (a DeepLog AST) via
a :class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`. It deals
only with plain Prolog: resolution, the logical connectives, builtins, and one
semantics-free extension concept -- *open* (leaf) predicates. A predicate is
either *defined* (has clauses -> resolve them; a deterministic fact contributes
``true``) or *open* (its facts are leaves -> emit an atom). Any meaning attached
to those leaves (probabilities, annotated disjunctions, ...) is the caller's
concern, applied after grounding.

This interface is deliberately Prolog-specific (a Prolog ``Program``, functor /
arity builtins, ``(functor, arity)`` open predicates); it is *not* a universal
grounder base. A different grounding front-end (e.g. a first-order-logic
grounder) would define its own interface -- the only thing every grounder shares
is the semantics-free proof structure it drives through a
:class:`~deeplog.grounding.ProofBuilder`.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING
from typing import TypeVar

from deeplog.symbol import Symbol


if TYPE_CHECKING:
    from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory

    from ..builder import ProofBuilder
    from .program import Program


T = TypeVar("T")
#: A builtin enumerates answer substitutions for a goal's arguments.
type Builtin = Callable[..., Iterable[dict[Symbol, Symbol]]]
#: The set of open (leaf) predicates, keyed by ``(functor, arity)``.
type OpenPredicates = AbstractSet[tuple[str, int]]


class UnknownPredicateException(Exception):
    """Raised when a goal's predicate has no clauses, builtin, or open declaration."""


class PrologGrounder(ABC):
    """Abstract plain-Prolog grounder: proves a goal to a per-answer proof formula."""

    @abstractmethod
    def ground(
        self,
        program: Program,
        goal: Symbol,
        factory: DeepLogFormulaFactory[T] | ProofBuilder[T],
        open_predicates: OpenPredicates = frozenset(),
    ) -> dict[Symbol, T]:
        """Prove ``goal`` in ``program`` and return one proof formula per ground answer.

        ``factory`` receives the proof structure (``and`` / ``or`` / ``not`` /
        atoms): either a raw :class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`
        (wrapped internally in a :class:`~deeplog.grounding.ProofBuilder`) or a
        ``ProofBuilder`` shared across several ``ground`` calls so their leaves
        co-reside. ``open_predicates`` names the predicates whose facts become
        leaf atoms rather than resolving to ``true``.
        """

    @abstractmethod
    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register an additional builtin predicate."""

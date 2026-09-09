#  Copyright (c) 2024-2026. KU Leuven
"""Abstract factory for building DeepLog formulas.

Concrete implementations live alongside (e.g. ``symbolic_factory.py``,
``deeplogmodulefactory/``).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar

from ..symbol import Symbol


if TYPE_CHECKING:
    from .ast import CircuitNode


T = TypeVar("T")


class DeepLogFormulaFactory[T](ABC):
    """The fold algebra over the formula AST.

    Each method is an *eliminator* of one formula-AST node kind: it receives the
    already-built results (carrier type ``T``) of that node's children and returns
    a ``T``. It does not operate on AST nodes directly — that is the job of
    :func:`~deeplog.formula.ast.fold`, which drives an instance of this class over
    a materialized tree.

    A :class:`~deeplog.formula.ast.CircuitNode` lump is graph-backed; the
    :attr:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.lowers_circuit_children`
    flag decides whether the fold descends its boundary children through the
    ordinary eliminators (lowering algebras) or splices it in opaquely
    (builders / interpreters).

    Concrete algebras choose ``T``: ``str`` for
    ``SymbolicFormulaFactory`` (→ text), ``CircuitNode | DeepLogModule`` for
    ``DeepLogModuleFactory`` (→ a circuit-backed module).
    """

    #: Whether :func:`~deeplog.formula.ast.fold` should descend a
    #: :class:`~deeplog.formula.ast.CircuitNode`'s boundary children (its leaf /
    #: cast AST view) through the ordinary eliminators and hand them to
    #: :meth:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.embed_circuit`,
    #: instead of splicing the lump in as an opaque leaf. A lowering algebra that
    #: turns leaves into modules sets this; the circuit builder and text
    #: interpreter leave it off (descending would rebuild circuit structure / has
    #: no surface syntax).
    lowers_circuit_children: bool = False

    @abstractmethod
    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[T],
        child: T,
    ) -> T:
        """Build an aggregation node over ``binders`` applying ``operation`` to ``child``."""

    @abstractmethod
    def create_transformation(self, structure: str, child: T) -> T:
        """Build a structure-conversion node that maps ``child`` to ``structure``."""

    @abstractmethod
    def create_binary_node(self, operator: str, lhs: T, rhs: T) -> T:
        """Combine ``lhs`` and ``rhs`` with a binary operator."""

    @abstractmethod
    def create_unary_node(self, operator: str, operand: T) -> T:
        """Apply a unary operator to ``operand``."""

    @abstractmethod
    def create_atom(self, atom: Symbol) -> T:
        """Build an atomic node."""

    @abstractmethod
    def embed_circuit(self, node: CircuitNode, children: tuple[T, ...] = ()) -> T:
        """Eliminate a :class:`~deeplog.formula.ast.CircuitNode` lump.

        When
        :attr:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.lowers_circuit_children`
        is set, ``children`` holds the fold results of the lump's boundary view
        (:attr:`~deeplog.formula.ast.CircuitNode.children`) - one per leaf /
        cast, already turned into ``T`` by ``create_atom`` /
        ``create_transformation`` - and this method composes them with the
        lump's compiled interior. When the flag is off, ``children`` is empty
        and the lump is spliced in verbatim (the circuit builder returns it;
        text interpreters reject it, since a compiled lump has no surface
        syntax).
        """

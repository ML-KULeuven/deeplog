#  Copyright (c) 2024-2026. KU Leuven
"""Abstract factory for building DeepLog formulas.

Concrete implementations live alongside (e.g. ``symbolic_factory.py``,
``circuit_factory.py``).
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

    Concrete algebras choose ``T``: ``str`` for ``SymbolicFormulaFactory``
    (→ text), ``FormulaNode`` for ``CircuitFactory`` (→ circuit lumps).
    """

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
    def embed_circuit(self, node: CircuitNode) -> T:
        """Eliminate a :class:`~deeplog.formula.ast.CircuitNode` lump.

        A lump is a compiled region of a circuit, so it is spliced in as it is:
        the circuit builder returns it, and text interpreters reject it, since a
        compiled lump has no surface syntax.
        """

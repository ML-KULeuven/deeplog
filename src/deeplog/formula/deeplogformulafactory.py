#  Copyright (c) 2024-2026. KU Leuven
"""Abstract factory for building DeepLog formulas.

Concrete implementations live alongside (e.g. ``symbolic_factory.py``,
``deeplogmodulefactory/``).
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from ..symbol import Symbol


T = TypeVar("T")


class DeepLogFormulaFactory[T](ABC):
    """Abstract factory for instantiating a DeepLog deeplogfactory."""

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

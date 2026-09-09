#  Copyright (c) 2024-2026. KU Leuven
"""AST-emitting concrete formula factory.

``AstFactory`` is the identity interpreter of the formula signature: each
eliminator is the constructor of the node kind it eliminates, so
``fold(node, AstFactory())`` reproduces ``node``. It is what a producer driven by
a :class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory` — a
grounder, a DIMACS reader — builds through to obtain a formula it can inspect,
rewrite and print, rather than one already lowered to text or to a circuit.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..symbol import Symbol
from .ast import Aggregation
from .ast import Atom
from .ast import BinaryOp
from .ast import CircuitNode
from .ast import FormulaNode
from .ast import Transformation
from .ast import UnaryOp
from .deeplogformulafactory import DeepLogFormulaFactory


class AstFactory(DeepLogFormulaFactory[FormulaNode]):
    """Concrete factory that materializes the formula AST."""

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[FormulaNode],
        child: FormulaNode,
    ) -> FormulaNode:
        """Build an :class:`~deeplog.formula.ast.Aggregation` node."""
        return Aggregation(operation, tuple(binders), tuple(params), child)

    def create_transformation(self, structure: str, child: FormulaNode) -> FormulaNode:
        """Build a :class:`~deeplog.formula.ast.Transformation` node."""
        return Transformation(structure, child)

    def create_binary_node(
        self, operator: str, lhs: FormulaNode, rhs: FormulaNode
    ) -> FormulaNode:
        """Build a :class:`~deeplog.formula.ast.BinaryOp` node."""
        return BinaryOp(operator, lhs, rhs)

    def create_unary_node(self, operator: str, operand: FormulaNode) -> FormulaNode:
        """Build a :class:`~deeplog.formula.ast.UnaryOp` node."""
        return UnaryOp(operator, operand)

    def create_atom(self, atom: Symbol) -> FormulaNode:
        """Build an :class:`~deeplog.formula.ast.Atom` leaf."""
        return Atom(atom)

    def embed_circuit(
        self, node: CircuitNode, children: tuple[FormulaNode, ...] = ()
    ) -> FormulaNode:
        """Splice a compiled lump in verbatim — it is already an AST node."""
        return node

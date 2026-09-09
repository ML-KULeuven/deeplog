#  Copyright (c) 2024-2026. KU Leuven
"""Helpers for tests that assert on a formula AST."""

from collections.abc import Iterator

from deeplog.formula import Atom
from deeplog.formula import BinaryOp
from deeplog.formula import FormulaNode
from deeplog.symbol import parse_symbol


def leaf(text: str) -> Atom:
    """The boolean leaf atom for the ground atom written as ``text``."""
    return Atom(("_", parse_symbol(text), ("boolean",)))


def operands(node: FormulaNode, operator: str) -> Iterator[FormulaNode]:
    """Yield the operands of an ``operator`` chain, flattening nested ones.

    An associative operator can be nested either way round and two producers of
    the same formula need not agree on which; this is the view in which they do.
    """
    if isinstance(node, BinaryOp) and node.operator == operator:
        yield from operands(node.lhs, operator)
        yield from operands(node.rhs, operator)
    else:
        yield node

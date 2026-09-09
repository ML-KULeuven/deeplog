#  Copyright (c) 2024-2026. KU Leuven
"""String-emitting concrete formula factory.

``SymbolicFormulaFactory`` produces strings that round-trip through
:func:`~deeplog.formula.text_parser_lark.parse_formula`, so the symbolic
representation and the text representation are the same artefact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..symbol import Symbol
from ..symbol import structure_of
from ..symbol import symbol_to_str
from ..symbol import without_structure
from .deeplogformulafactory import DeepLogFormulaFactory


if TYPE_CHECKING:
    from .ast import CircuitNode


def _needs_wrap(s: str) -> bool:
    """Return True if ``s`` has a space outside all parentheses.

    Such strings need to be parenthesised when used as an operand so the
    parser does not mis-associate them.
    """
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == " " and depth == 0:
            return True
    return False


class SymbolicFormulaFactory(DeepLogFormulaFactory[str]):
    """Concrete factory that emits formula text strings.

    Every method returns a string that is valid input for
    :func:`~deeplog.formula.text_parser_lark.parse_formula`, so the
    symbolic representation and the text representation are one and the same.
    That is also why it rejects a compiled lump: a handle onto a circuit parses
    as an atom, not as the formula the lump stands for. Rendering one is what
    ``str`` on an AST node does, through a subclass of this factory.
    """

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[str],
        child: str,
    ) -> str:
        """Encode an aggregation as ``operation(binders; params): child``."""
        binder_text = ", ".join(symbol_to_str(b) for b in binders)
        params_section = f"; {', '.join(params)}" if params else ""
        return f"{operation}({binder_text}{params_section}): {child}"

    def create_transformation(self, structure: str, child: str) -> str:
        """Wrap ``child`` with a structure transformation: ``(child)_structure``."""
        return f"({child})_{structure}"

    def create_binary_node(self, operator: str, lhs: str, rhs: str) -> str:
        """Return ``lhs operator rhs``, parenthesising operands that need it."""
        lhs = f"({lhs})" if _needs_wrap(lhs) else lhs
        rhs = f"({rhs})" if _needs_wrap(rhs) else rhs
        return f"{lhs} {operator} {rhs}"

    def create_unary_node(self, operator: str, operand: str) -> str:
        """Return ``operator operand``."""
        return f"{operator} {operand}"

    def create_atom(self, atom: Symbol) -> str:
        """Render a leaf atom as ``symbol_text_structure``, or bare if untagged."""
        structure = structure_of(atom)
        if structure is None:
            return symbol_to_str(atom)
        return f"{symbol_to_str(without_structure(atom))}_{structure}"

    def embed_circuit(self, node: CircuitNode, children: tuple[str, ...] = ()) -> str:
        """Reject a compiled lump — it has no surface syntax to render.

        Lumps are emitted only on the compute path (the DeepProbLog engine) and
        never reach the text interpreter, so this never fires in practice.
        """
        raise NotImplementedError(
            "a compiled CircuitNode lump cannot be rendered as formula text"
        )

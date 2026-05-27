#  Copyright (c) 2024-2026. KU Leuven
"""String-emitting concrete formula factory.

``SymbolicFormulaFactory`` produces strings that round-trip through
:func:`~deeplog.formula.text_parser_lark.parse_formula`, so the symbolic
representation and the text representation are the same artefact.
"""

from collections.abc import Sequence

from ..algebraic import Algebra
from ..algebraic import structure_registry
from ..symbol import Symbol
from ..symbol import symbol_to_str
from .deeplogformulafactory import DeepLogFormulaFactory


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
        """Return ``lhs operator rhs``, parenthesising operands that need it.

        Neutral elements (additive zero, multiplicative one) are simplified
        away eagerly so that downstream code never sees redundant terms.
        """
        for struct_name, algebra in structure_registry.items():
            if isinstance(algebra, Algebra):
                zero = f"{symbol_to_str(algebra.zero)}_{struct_name}"
                one = f"{symbol_to_str(algebra.one)}_{struct_name}"
                if operator == algebra.sum:
                    if lhs == zero:
                        return rhs
                    if rhs == zero:
                        return lhs
                if operator == algebra.product:
                    if lhs == one:
                        return rhs
                    if rhs == one:
                        return lhs
        lhs = f"({lhs})" if _needs_wrap(lhs) else lhs
        rhs = f"({rhs})" if _needs_wrap(rhs) else rhs
        return f"{lhs} {operator} {rhs}"

    def create_unary_node(self, operator: str, operand: str) -> str:
        """Return ``operator operand``."""
        return f"{operator} {operand}"

    def create_atom(self, atom: Symbol) -> str:
        """Render a leaf atom as ``symbol_text_structure``."""
        _, symbol, (structure,) = atom
        return f"{symbol_to_str(symbol)}_{structure}"

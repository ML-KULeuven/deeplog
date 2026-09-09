#  Copyright (c) 2024-2026. KU Leuven
"""Parse DeepLog formulas using the Lark parser."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

from lark import Lark
from lark import Token
from lark import Tree

from ..module import DeepLogModule
from ..symbol import parse_symbol
from .ast import Aggregation
from .ast import Atom
from .ast import BinaryOp
from .ast import FormulaNode
from .ast import Transformation
from .ast import UnaryOp
from .ast import fold
from .ast import hash_cons
from .deeplogformulafactory import DeepLogFormulaFactory
from .deeplogmodulefactory import DeepLogModuleFactory
from .passes import DEFAULT_PASSES


_GRAMMAR = r"""
%import common.WS
%ignore WS
%ignore COMMENT
COMMENT: /#[^\n]*/

start: deeplogfactory

?deeplogfactory: aggregation
        | binary

aggregation: IDENT "(" binders (";" params)? ")" ":" deeplogfactory

binders: IDENT ("," IDENT)*
params: deeplogfactory ("," deeplogfactory)*

binary: unary (IDENT unary)*

?unary: prefix
      | transformation
      | grouped
      | leaf

prefix: IDENT unary
grouped: "(" deeplogfactory ")"
transformation: grouped STRUCT_SUFFIX
leaf: SYMBOL STRUCT_SUFFIX

// SYMBOL/STRUCT_SUFFIX outrank the generic IDENT so a bare-atom leaf such as
// `x2_fuzzy` is lexed as SYMBOL `x2` + STRUCT_SUFFIX `_fuzzy`, not as the single
// operator IDENT `x2_fuzzy`. A leaf always ends in a `_<struct>` suffix and an
// operator/aggregation/binder IDENT never does, so this only resolves the
// genuinely-ambiguous leaf-vs-operator cases (previously the Earley dynamic
// lexer could read a leaf in operator position -> "Unknown operator 'x2_fuzzy'").
SYMBOL.2: /[^_()\s]+(\([^\)]+\))?/
STRUCT_SUFFIX.2: /_[A-Za-z_][A-Za-z0-9_]*/
IDENT: /[A-Za-z_][A-Za-z0-9_]*/
"""

_STRUCTURE_ALIASES = {
    "b": "boolean",
    "p": "probability",
}

_LARK = Lark(_GRAMMAR, parser="earley", maybe_placeholders=False)


def parse_formula_to_ast(text: str) -> FormulaNode:
    """Parse ``text`` into the materialized formula AST (a canonical DAG).

    Equal subformulas are interned to one object via :func:`hash_cons`, so the
    AST is a value-keyed DAG rather than a tree. This is a transparent efficiency
    move — ``fold`` re-walks the DAG identically to the tree, so behaviour (and
    the compiled circuit) is unchanged.
    """
    tree: Tree = _LARK.parse(text)
    return hash_cons(_evaluate_tree(tree), {})


def parse_formula[T](text: str, factory: DeepLogFormulaFactory[T]) -> T:
    """Parse ``text`` and fold it through ``factory`` (the AST interpreter)."""
    return fold(parse_formula_to_ast(text), factory)


def _node_name(node):
    if isinstance(node, Tree):
        data = node.data
        return data.value if isinstance(data, Token) else data
    raise ValueError("Expected Tree node")


def _token_value(token: Token) -> str:
    return token.value


def _structure_name(token: Token) -> str:
    name: str = token.value[1:]
    return _STRUCTURE_ALIASES.get(name, name)


def _evaluate_tree(node) -> FormulaNode:
    """Build the formula AST from a Lark parse tree (the grammar→node mapping)."""
    name = _node_name(node)
    children = node.children
    if name == "start":
        return _evaluate_tree(children[0])
    if name == "aggregation":
        op = children[0]
        binders_node = children[1]
        if len(children) == 3:
            params_node = None
            child = children[2]
        else:
            params_node = children[2]
            child = children[3]
        binders = tuple(
            parse_symbol(_token_value(token)) for token in binders_node.children
        )
        params = (
            ()
            if params_node is None
            else tuple(_evaluate_tree(param) for param in params_node.children)
        )
        return Aggregation(_token_value(op), binders, params, _evaluate_tree(child))
    if name == "binary":
        iter_children = iter(children)
        result = _evaluate_tree(next(iter_children))
        for op, operand in zip(iter_children, iter_children, strict=True):
            result = BinaryOp(_token_value(op), result, _evaluate_tree(operand))
        return result
    if name == "prefix":
        operator, operand = children
        return UnaryOp(_token_value(operator), _evaluate_tree(operand))
    if name == "grouped":
        return _evaluate_tree(children[0])
    if name == "transformation":
        child, structure = children
        return Transformation(_structure_name(structure), _evaluate_tree(child))
    if name == "leaf":
        symbol_token, structure = children
        return Atom(
            (
                "_",
                parse_symbol(_token_value(symbol_token)),
                (_structure_name(structure),),
            )
        )
    raise ValueError(f"Unexpected node {name}")


def parse_formula_to_module(
    text: str,
    factory: DeepLogModuleFactory | None = None,
    *,
    passes: Sequence[Callable[[FormulaNode], FormulaNode]] | None = None,
) -> DeepLogModule:
    """
    Parse ``text`` and convert it into a :class:`DeepLogModule`.

    The pipeline is ``parse → AST → passes → fold(factory)``. ``passes`` defaults
    to :data:`~deeplog.formula.passes.DEFAULT_PASSES` (currently conservative
    expectation/WMC recognition); pass ``passes=()`` to fold the formula verbatim.
    When no ``factory`` is provided, an empty
    :class:`DeepLogModuleFactory` is instantiated.
    """
    factory = factory or DeepLogModuleFactory()
    ast = parse_formula_to_ast(text)
    for rewrite in DEFAULT_PASSES if passes is None else passes:
        ast = rewrite(ast)
    return factory.compile(ast)

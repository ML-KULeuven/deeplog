#  Copyright (c) 2024-2026. KU Leuven
"""Parse DeepLog formulas using the Lark parser."""

from __future__ import annotations

from lark import Lark
from lark import Token
from lark import Tree

from ..module.deeplog_module import DeepLogModule
from ..symbol import Symbol
from ..symbol import parse_symbol
from .ast import Aggregation
from .ast import Atom
from .ast import BinaryOp
from .ast import FormulaNode
from .ast import Transformation
from .ast import UnaryOp
from .ast import hash_cons
from .lowering.compiler import Compiler


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

prefix.-1: IDENT unary
grouped: "(" deeplogfactory ")"
transformation: grouped STRUCT_SUFFIX
leaf: symbol STRUCT_SUFFIX

// A symbol is the text it spans, read by parse_symbol; only its parentheses
// are parsed here, so that its arguments may nest.
symbol: SYMBOL
      | FUNCTOR _arguments ")"
_arguments: (ARGUMENT_TEXT | "(" _arguments ")")+

// Where text reads more than one way, the reading with the fewest prefix
// operators wins: a bare leaf such as `x2_fuzzy` is not the operator
// `x2_fuzzy`, and a functor written against its arguments is one atom, as in
// Prolog: `q(x_bar)_boolean` is an atom, `q (x_bar)_boolean` the operator `q`
// over a cast.
SYMBOL: /[^_()\s]+/
FUNCTOR: /[^_()\s]+\(/
ARGUMENT_TEXT: /[^()]+/
STRUCT_SUFFIX: /_[A-Za-z_][A-Za-z0-9_]*/
IDENT: /[A-Za-z_][A-Za-z0-9_]*/
"""

_STRUCTURE_ALIASES = {
    "b": "boolean",
    "p": "probability",
}

_LARK = Lark(
    _GRAMMAR, parser="earley", maybe_placeholders=False, propagate_positions=True
)


def parse_formula(text: str) -> FormulaNode:
    """Parse ``text`` into a formula.

    Equal subformulas are interned to one object via :func:`hash_cons`, so the
    formula is a value-keyed DAG rather than a tree.
    """
    tree: Tree = _LARK.parse(text)
    return hash_cons(_evaluate_tree(tree, text), {})


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


def _evaluate_tree(root: Tree, text: str) -> FormulaNode:
    """Build the formula AST from a Lark parse tree (the grammar→node mapping)."""
    values: dict[int, object] = {}
    # An explicit stack rather than recursion, as in fold: text can nest thousands
    # deep. Each tree is built from its children's values, its tokens as they are.
    stack: list[tuple[Tree, bool]] = [(root, False)]
    while stack:
        node, ready = stack.pop()
        if not ready:
            stack.append((node, True))
            stack.extend(
                (child, False) for child in node.children if isinstance(child, Tree)
            )
            continue
        args = [
            values.pop(id(child)) if isinstance(child, Tree) else child
            for child in node.children
        ]
        values[id(node)] = _build(node, args, text)
    return values[id(root)]  # pyright: ignore[reportReturnType]


def _build(node: Tree, args: list, text: str) -> object:
    """The value of the parse tree ``node``, given its children's values ``args``."""
    name = _node_name(node)
    if name in ("start", "grouped"):
        return args[0]
    if name == "aggregation":
        op, binders, *rest = args
        params, child = rest if len(rest) == 2 else ((), rest[0])
        return Aggregation(_token_value(op), binders, params, child)
    if name == "binders":
        return tuple(parse_symbol(_token_value(token)) for token in args)
    if name == "params":
        return tuple(args)
    if name == "binary":
        result, *rest = args
        for op, operand in zip(rest[::2], rest[1::2], strict=True):
            result = BinaryOp(_token_value(op), result, operand)
        return result
    if name == "prefix":
        operator, operand = args
        return UnaryOp(_token_value(operator), operand)
    if name == "transformation":
        child, structure = args
        return Transformation(_structure_name(structure), child)
    if name == "leaf":
        symbol, structure = args
        return Atom(("_", symbol, (_structure_name(structure),)))
    if name == "symbol":
        return parse_symbol(text[node.meta.start_pos : node.meta.end_pos])
    raise ValueError(f"Unexpected node {name}")


def parse_formula_to_module(
    text: str, compiler: Compiler | None = None, *, name: str | Symbol | None = None
) -> DeepLogModule:
    """Parse ``text`` and compile it: ``compiler.compile(parse_formula(text))``.

    ``compiler`` defaults to a :class:`~deeplog.formula.lowering.compiler.Compiler`
    with the default builders. With ``name``, the module's column is named by it,
    as ``compiler.compile({name: parse_formula(text)})`` names it.
    """
    formula = parse_formula(text)
    compiler = compiler or Compiler()
    if name is None:
        return compiler.compile(formula)
    return compiler.compile({name: formula})

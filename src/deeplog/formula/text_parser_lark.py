#  Copyright (c) 2024-2026. KU Leuven
"""Parse DeepLog formulas using the Lark parser."""

from __future__ import annotations

from typing import Any
from typing import cast

from lark import Lark
from lark import Token
from lark import Tree

from ..module import DeepLogModule
from ..symbol import parse_symbol
from .deeplogformulafactory import DeepLogFormulaFactory
from .deeplogmodulefactory import DeepLogModuleFactory


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

SYMBOL: /[^_()\s]+(\([^\)]+\))?/
IDENT: /[A-Za-z_][A-Za-z0-9_]*/
STRUCT_SUFFIX: /_[A-Za-z_][A-Za-z0-9_]*/
"""

_STRUCTURE_ALIASES = {
    "b": "boolean",
    "p": "probability",
}

_LARK = Lark(_GRAMMAR, parser="earley", maybe_placeholders=False)


def parse_formula[T](text: str, factory: DeepLogFormulaFactory[T]) -> T:
    """Parse ``text`` using Lark and emit nodes via ``factory``."""
    tree: Tree = _LARK.parse(text)
    return cast(T, _evaluate_tree(tree, factory))


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


def _evaluate_tree(node, factory: DeepLogFormulaFactory[Any]):
    name = _node_name(node)
    children = node.children
    if name == "start":
        return _evaluate_tree(children[0], factory)
    if name == "aggregation":
        op = children[0]
        binders_node = children[1]
        if len(children) == 3:
            params_node = None
            child = children[2]
        else:
            params_node = children[2]
            child = children[3]
        binders = [parse_symbol(_token_value(token)) for token in binders_node.children]
        params = (
            []
            if params_node is None
            else [_evaluate_tree(param, factory) for param in params_node.children]
        )
        return factory.create_aggregation(
            _token_value(op), binders, params, _evaluate_tree(child, factory)
        )
    if name == "binary":
        iter_children = iter(children)
        result = _evaluate_tree(next(iter_children), factory)
        for op, operand in zip(iter_children, iter_children, strict=True):
            result = factory.create_binary_node(
                _token_value(op), result, _evaluate_tree(operand, factory)
            )
        return result
    if name == "prefix":
        operator, operand = children
        return factory.create_unary_node(
            _token_value(operator), _evaluate_tree(operand, factory)
        )
    if name == "grouped":
        return _evaluate_tree(children[0], factory)
    if name == "transformation":
        child, structure = children
        return factory.create_transformation(
            _structure_name(structure), _evaluate_tree(child, factory)
        )
    if name == "leaf":
        symbol_token, structure = children
        return factory.create_atom(
            (
                "_",
                parse_symbol(_token_value(symbol_token)),
                (_structure_name(structure),),
            )
        )
    if isinstance(node, Token):
        return node
    raise ValueError(f"Unexpected node {name}")


def parse_formula_to_module(
    text: str, factory: DeepLogModuleFactory | None = None
) -> DeepLogModule:
    """
    Parse ``text`` and immediately convert it into a :class:`DeepLogModule`.

    When no ``factory`` is provided, an empty :class:`DeepLogModuleFactory`
    is instantiated.
    """
    factory = factory or DeepLogModuleFactory()
    return parse_formula(text, factory).to_module()

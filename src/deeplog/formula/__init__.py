#  Copyright (c) 2024-2026. KU Leuven
"""A module containing everything related to operating on DeepLog formulas"""

from ..symbol import structure_of
from ..symbol import with_structure
from .ast import Aggregation
from .ast import Atom
from .ast import BinaryOp
from .ast import CircuitNode
from .ast import FormulaNode
from .ast import Transformation
from .ast import UnaryOp
from .ast import children
from .ast import fold
from .ast import map_children
from .ast_factory import AstFactory
from .circuit_node import to_module
from .circuit_node import transform_nodes
from .deeplogformulafactory import DeepLogFormulaFactory
from .deeplogmodulefactory import AggregationBuilder
from .deeplogmodulefactory import AtomBuilder
from .deeplogmodulefactory import DeepLogModuleFactory
from .deeplogmodulefactory import TransformationBuilder
from .deeplogmodulefactory import lower_circuit_nodes
from .dimacs_parser import parse_dimacs_cnf
from .passes import recognize_expectation
from .passes import recognize_posterior
from .predicates import Arguments
from .predicates import EqualityPredicate
from .predicates import LogProbabilityPredicate
from .predicates import NetworkPredicate
from .predicates import Predicate
from .predicates import ProbabilityPredicate
from .predicates import SumsPredicate
from .predicates import get_network_predicate
from .symbolic_factory import SymbolicFormulaFactory
from .text_parser_lark import parse_formula
from .text_parser_lark import parse_formula_to_ast
from .text_parser_lark import parse_formula_to_module


__all__ = [
    "parse_formula",
    "parse_formula_to_ast",
    "parse_formula_to_module",
    "parse_dimacs_cnf",
    "DeepLogFormulaFactory",
    "DeepLogModuleFactory",
    "AggregationBuilder",
    "AtomBuilder",
    "TransformationBuilder",
    "lower_circuit_nodes",
    "SymbolicFormulaFactory",
    "AstFactory",
    "FormulaNode",
    "Atom",
    "UnaryOp",
    "BinaryOp",
    "Transformation",
    "Aggregation",
    "CircuitNode",
    "fold",
    "children",
    "map_children",
    "to_module",
    "transform_nodes",
    "recognize_expectation",
    "recognize_posterior",
    "Arguments",
    "EqualityPredicate",
    "LogProbabilityPredicate",
    "Predicate",
    "NetworkPredicate",
    "ProbabilityPredicate",
    "SumsPredicate",
    "get_network_predicate",
    "structure_of",
    "with_structure",
]

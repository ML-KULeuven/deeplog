#  Copyright (c) 2024-2026. KU Leuven
"""DeepLog's public API: stating a model, compiling it, running it and extending it.

The other public namespaces are :mod:`deeplog.circuit`, for algebraic circuits,
and one package per grounder under :mod:`deeplog.grounding`.
"""

from .algebraic import BOOLEAN
from .algebraic import LOGPROBABILITY
from .algebraic import MPE
from .algebraic import PROBABILITY
from .algebraic import REAL
from .algebraic import Algebra
from .algebraic import AlgebraicStructure
from .algebraic import Semifield
from .algebraic import Semiring
from .formula.ast import Aggregation
from .formula.ast import Atom
from .formula.ast import BinaryOp
from .formula.ast import FormulaNode
from .formula.ast import Transformation
from .formula.ast import UnaryOp
from .formula.dimacs_parser import parse_dimacs_cnf
from .formula.lowering.builder_protocols import AggregationBuilder
from .formula.lowering.builder_protocols import AtomBuilder
from .formula.lowering.builder_protocols import DeclaresDomains
from .formula.lowering.builder_protocols import TransformationBuilder
from .formula.lowering.compiler import Compiler
from .formula.lowering.expectation import enumeration
from .formula.lowering.expectation import sampling
from .formula.lowering.expectation import score_function
from .formula.lowering.expectation import weighted_model_count
from .formula.lowering.lowering import Lowering
from .formula.lowering.reduction import reduction
from .formula.predicates.builtin_predicates import EqualityPredicate
from .formula.predicates.builtin_predicates import LogProbabilityPredicate
from .formula.predicates.builtin_predicates import NetworkPredicate
from .formula.predicates.builtin_predicates import ProbabilityPredicate
from .formula.predicates.predicate import Predicate
from .formula.text_parser_lark import parse_formula
from .formula.text_parser_lark import parse_formula_to_module
from .module.deeplog_module import DeepLogModule
from .module.module_circuit import compose_modules
from .module.reshape import TransformationNotPossible
from .module.reshape import reshape
from .module.sequential import Sequential
from .module.wrappers import WrappedModule
from .shape import Shape
from .shape import ShapeMismatchException
from .shape import SymbolDict
from .shape import SymTensor
from .shape import SymTensorLike
from .shape import get_all_symbols
from .shape import to_dict
from .symbol import Symbol
from .symbol import apply_substitution
from .symbol import calculate_mgu
from .symbol import flatten_symbol
from .symbol import get_predicate
from .symbol import get_term_variables
from .symbol import is_variable
from .symbol import parse_symbol
from .symbol import structure_of
from .symbol import symbol_to_pretty_string
from .symbol import symbol_to_str
from .symbol import to_symbol
from .symbol import with_structure
from .symbol import without_structure
from .variable import OPEN
from .variable import Domain
from .variable import Variable
from .variable import VariableAtoms


__all__ = [
    # Symbols
    "Symbol",
    "parse_symbol",
    "symbol_to_str",
    "symbol_to_pretty_string",
    "with_structure",
    "structure_of",
    "without_structure",
    # Terms
    "is_variable",
    "get_predicate",
    "get_term_variables",
    "flatten_symbol",
    "to_symbol",
    "apply_substitution",
    "calculate_mgu",
    # Formulas
    "FormulaNode",
    "Atom",
    "UnaryOp",
    "BinaryOp",
    "Transformation",
    "Aggregation",
    "parse_formula",
    "parse_dimacs_cnf",
    # Algebras
    "AlgebraicStructure",
    "Semiring",
    "Semifield",
    "Algebra",
    "BOOLEAN",
    "PROBABILITY",
    "LOGPROBABILITY",
    "REAL",
    "MPE",
    # What atoms mean
    "Predicate",
    "NetworkPredicate",
    "EqualityPredicate",
    "ProbabilityPredicate",
    "LogProbabilityPredicate",
    "AtomBuilder",
    "DeclaresDomains",
    # Domains and variables
    "Domain",
    "OPEN",
    "Variable",
    "VariableAtoms",
    # Aggregations and casts
    "AggregationBuilder",
    "TransformationBuilder",
    "Lowering",
    "reduction",
    "weighted_model_count",
    "enumeration",
    "sampling",
    "score_function",
    # Compiling
    "Compiler",
    "parse_formula_to_module",
    # Modules
    "DeepLogModule",
    "SymTensor",
    "SymTensorLike",
    "Shape",
    "SymbolDict",
    "reshape",
    "to_dict",
    "get_all_symbols",
    "WrappedModule",
    "compose_modules",
    "Sequential",
    "ShapeMismatchException",
    "TransformationNotPossible",
]

#  Copyright (c) 2024-2026. KU Leuven
"""The plain-Prolog grounder: program representation, parser, and provers."""

from .grounder import Builtin
from .grounder import OpenPredicates
from .grounder import PrologGrounder
from .grounder import UnknownPredicateException
from .janus import JANUS_AVAILABLE
from .janus import JanusGrounder
from .janus import JanusNotAvailableException
from .janus import JanusProver
from .parser import str_to_rule
from .parser import str_to_rules
from .parser import symbol_to_prolog_str
from .program import Program
from .program import RuleType
from .program import create_fact
from .program import create_query
from .program import create_rule
from .program import get_constraint_body
from .program import is_constraint
from .program import is_fact
from .program import is_query
from .simple import SimpleGrounder
from .unify import calculate_mgu


__all__ = [
    "Program",
    "RuleType",
    "create_rule",
    "create_query",
    "create_fact",
    "is_fact",
    "is_query",
    "is_constraint",
    "get_constraint_body",
    "str_to_rule",
    "str_to_rules",
    "symbol_to_prolog_str",
    "calculate_mgu",
    "PrologGrounder",
    "Builtin",
    "OpenPredicates",
    "UnknownPredicateException",
    "SimpleGrounder",
    "JanusGrounder",
    "JanusProver",
    "JanusNotAvailableException",
    "JANUS_AVAILABLE",
]

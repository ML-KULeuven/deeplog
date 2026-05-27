#  Copyright (c) 2024-2026. KU Leuven
"""Built-in predicates for DeepLog formulas."""

from .builtin_predicates import EqualityPredicate
from .builtin_predicates import LogProbabilityPredicate
from .builtin_predicates import ProbabilityPredicate
from .builtin_predicates import SumsPredicate
from .builtin_predicates import get_network_predicate
from .predicate import Predicate


__all__ = [
    "Predicate",
    "SumsPredicate",
    "EqualityPredicate",
    "ProbabilityPredicate",
    "LogProbabilityPredicate",
    "get_network_predicate",
]

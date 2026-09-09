#  Copyright (c) 2024-2026. KU Leuven
"""Built-in predicates for DeepLog formulas."""

from .builtin_predicates import EqualityPredicate as EqualityPredicate
from .builtin_predicates import LogProbabilityPredicate as LogProbabilityPredicate
from .builtin_predicates import NetworkPredicate as NetworkPredicate
from .builtin_predicates import ProbabilityPredicate as ProbabilityPredicate
from .builtin_predicates import SumsPredicate as SumsPredicate
from .builtin_predicates import get_network_predicate as get_network_predicate
from .predicate import Arguments as Arguments
from .predicate import Predicate as Predicate

#  Copyright (c) 2024-2026. KU Leuven
"""DeepProbLog system integration for DeepLog.

DeepProbLog layers probabilistic (and neural) semantics on top of a plain-Prolog
grounder (:mod:`deeplog.grounding.prolog`). A :class:`Solver` wraps a grounder
(:class:`~deeplog.grounding.prolog.SimpleGrounder` /
:class:`~deeplog.grounding.prolog.JanusGrounder`) and produces an
:class:`EngineResult`; :class:`KBestJanusGrounder` is the probability-guided
variant used directly. :func:`compile_to_module` turns a result into a
differentiable module. Like every system, it is outside DeepLog's stability
promise (:mod:`deeplog.systems`).
"""

from deeplog.grounding.prolog import Builtin
from deeplog.grounding.prolog import JanusGrounder
from deeplog.grounding.prolog import Program
from deeplog.grounding.prolog import PrologGrounder
from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog import SimpleGrounder
from deeplog.grounding.prolog import UnknownPredicateException
from deeplog.grounding.prolog import create_fact
from deeplog.grounding.prolog import create_query
from deeplog.grounding.prolog import create_rule
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_query

from .compile import compile_to_module
from .kbest.kbest import KBestJanusGrounder
from .kbest.predicate_evaluator import NeuralPredicateEvaluator
from .parser import create_labeled_fact
from .solver import EngineResult
from .solver import Solver


__all__ = [
    # Solving
    "Solver",
    "EngineResult",
    "compile_to_module",
    # Grounders (re-exported for convenience)
    "PrologGrounder",
    "SimpleGrounder",
    "JanusGrounder",
    "KBestJanusGrounder",
    "NeuralPredicateEvaluator",
    "Builtin",
    "UnknownPredicateException",
    # Program: types and construction
    "Program",
    "RuleType",
    "create_rule",
    "create_query",
    "create_fact",
    "create_labeled_fact",
    "is_query",
    "is_constraint",
    "get_constraint_body",
]

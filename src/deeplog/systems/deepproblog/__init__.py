#  Copyright (c) 2024-2026. KU Leuven
"""DeepProbLog system integration for DeepLog.

DeepProbLog layers probabilistic (and neural) semantics on top of a plain-Prolog
grounder (:mod:`deeplog.grounding`). A :class:`Solver` wraps a grounder
(:class:`~deeplog.grounding.SimpleGrounder` / :class:`~deeplog.grounding.JanusGrounder`)
and produces an :class:`EngineResult`; :class:`KBestJanusGrounder` is the
probability-guided variant used directly. :func:`compile_to_module` turns a result
into a differentiable module.
"""

from deeplog.grounding import JANUS_AVAILABLE
from deeplog.grounding import Builtin
from deeplog.grounding import JanusGrounder
from deeplog.grounding import Program
from deeplog.grounding import PrologGrounder
from deeplog.grounding import RuleType
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import UnknownPredicateException
from deeplog.grounding.prolog import create_fact
from deeplog.grounding.prolog import create_query
from deeplog.grounding.prolog import create_rule
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_query

from .compile import compile_to_module
from .kbest import KBestJanusGrounder
from .kbest import NeuralPredicateEvaluator
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
    "JANUS_AVAILABLE",
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

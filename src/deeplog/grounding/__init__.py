#  Copyright (c) 2024-2026. KU Leuven
"""First-class grounders: from a high-level logic language to a DeepLog formula AST.

A grounder proves a query in a program and emits its proof structure through a
:class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`. The one
piece every grounder shares is that semantics-free proof-building seam
(:class:`ProofBuilder`); the grounding interface itself is front-end specific.
This package currently provides the plain-Prolog grounder
(:mod:`deeplog.grounding.prolog`, exposing :class:`PrologGrounder` and its
:class:`SimpleGrounder` / :class:`JanusGrounder` implementations); it is free of
any probabilistic / neural / annotated-disjunction semantics, which callers layer
on top (see :mod:`deeplog.systems.deepproblog`).
"""

from .builder import ProofBuilder
from .prolog import JANUS_AVAILABLE
from .prolog import Builtin
from .prolog import JanusGrounder
from .prolog import JanusNotAvailableException
from .prolog import OpenPredicates
from .prolog import Program
from .prolog import PrologGrounder
from .prolog import RuleType
from .prolog import SimpleGrounder
from .prolog import UnknownPredicateException
from .prolog import str_to_rule
from .prolog import str_to_rules


__all__ = [
    "ProofBuilder",
    "PrologGrounder",
    "Builtin",
    "OpenPredicates",
    "UnknownPredicateException",
    "SimpleGrounder",
    "JanusGrounder",
    "JanusNotAvailableException",
    "JANUS_AVAILABLE",
    "Program",
    "RuleType",
    "str_to_rule",
    "str_to_rules",
]

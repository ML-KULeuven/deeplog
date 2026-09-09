#  Copyright (c) 2024-2026. KU Leuven
"""Plain-Prolog programs: rules, facts, queries and integrity constraints.

This module is deliberately free of any probabilistic / annotated-disjunction
vocabulary, and of parsing -- it is just the clause data type and its
constructors / accessors. A clause is a ``(functor, head, body)`` triple over the
universal :data:`~deeplog.symbol.Symbol` term type; the only kinds are rules
(``:-``), queries (``?-``) and integrity constraints (``:- body`` with an empty
head). Parsing plain-Prolog source text into these clauses lives next door in
:mod:`~deeplog.grounding.prolog.parser`.
"""

from deeplog.symbol import FalseSymbol
from deeplog.symbol import Symbol
from deeplog.symbol import TrueSymbol
from deeplog.symbol import to_symbol
from deeplog.util import foldr


#: A clause as a ``(functor, head, body)`` triple, the functor saying whether it
#: is a rule, a query or a constraint.
type RuleType = tuple[str, Symbol, Symbol]
#: The clauses a grounder proves against, in the order they were written.
type Program = tuple[RuleType, ...]


def create_rule(
    head_atoms: list[str] | list[Symbol],
    body_atoms: list[str] | list[Symbol],
    functor: str = ":-",
) -> RuleType:
    """Create a rule ``h1 ; ... ; hn <functor> b1 , ... , bm``.

    An empty head folds to :data:`~deeplog.symbol.FalseSymbol` and an empty body
    to :data:`~deeplog.symbol.TrueSymbol`, so facts (empty body) and constraints
    (empty head) are just special cases.
    """
    head: Symbol = (
        FalseSymbol
        if len(head_atoms) == 0
        else foldr(lambda x, y: (";", x, y), to_symbol(head_atoms))
    )
    body: Symbol = (
        TrueSymbol
        if len(body_atoms) == 0
        else foldr(lambda x, y: (",", x, y), to_symbol(body_atoms))
    )
    return functor, head, body


def is_rule(symbol: Symbol) -> bool:
    """Return true if ``symbol`` is a ``:-`` clause of arity 3."""
    return len(symbol) == 3 and symbol[0] == ":-"


def create_fact(atom: Symbol | str) -> RuleType:
    """Create a plain fact ``atom :- true``."""
    return create_rule([to_symbol(atom)], [])


def is_fact(symbol: Symbol) -> bool:
    """Return true if ``symbol`` is a rule with an empty (``true``) body."""
    return is_rule(symbol) and symbol[2] == TrueSymbol


def create_query(body_atoms: list[str] | list[Symbol]) -> RuleType:
    """Create a query ``false ?- b1 , ... , bm``."""
    return create_rule([], body_atoms, "?-")


def is_query(symbol: Symbol) -> bool:
    """Return true if ``symbol`` is a ``?-`` directive with a false head."""
    return len(symbol) == 3 and symbol[0] == "?-" and symbol[1] == FalseSymbol


def create_constraint(body_atoms: list[str] | list[Symbol]) -> RuleType:
    """Create an integrity constraint ``false :- b1 , ... , bm``."""
    return create_rule([], body_atoms)


def is_constraint(symbol: Symbol) -> bool:
    """Return true if ``symbol`` is a rule with a false head."""
    return is_rule(symbol) and symbol[1] == FalseSymbol


def get_constraint_body(rule: RuleType) -> Symbol:
    """Return the body of an integrity constraint ``:- body``."""
    return rule[2]

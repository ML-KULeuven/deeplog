#  Copyright (c) 2024-2026. KU Leuven
"""DeepProbLog surface syntax: the ``p :: a`` label.

Parsing is the grounder's (:func:`deeplog.grounding.str_to_rule`), which reads
``::`` as the binary term it is; this module gives that term its DeepProbLog
reading, building and taking apart the labeled atom ``("::", label, atom)``.
"""

from deeplog import Symbol
from deeplog import to_symbol
from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog import create_rule


def create_labeled_atom(atom: Symbol | str, label: Symbol | str | None) -> Symbol:
    """Return ``atom`` if ``label`` is None, else the labeled atom ``label :: atom``."""
    atom = to_symbol(atom)
    if label is None:
        return atom
    return "::", to_symbol(label), atom


def get_label(atom: Symbol) -> Symbol | None:
    """Return the label of a ``label :: atom`` term, or None if unlabeled."""
    if len(atom) == 3 and atom[0] == "::":
        return atom[1]
    return None


def get_atom(atom: Symbol) -> Symbol:
    """Return the bare atom of a ``label :: atom`` term, or ``atom`` itself."""
    if len(atom) == 3 and atom[0] == "::":
        return atom[2]
    return atom


def create_labeled_fact(atom: Symbol | str, label: Symbol | str | None) -> RuleType:
    """Create a (possibly labeled) fact ``label :: atom :- true``."""
    return create_rule([create_labeled_atom(atom, label)], [])


def get_fact_atom(fact: RuleType) -> Symbol:
    """Return the atom of a fact."""
    return get_atom(fact[1])


def get_fact_label(fact: RuleType) -> Symbol | None:
    """Return the label of a fact, or None if unlabeled."""
    return get_label(fact[1])

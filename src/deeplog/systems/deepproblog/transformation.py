#  Copyright (c) 2024-2026. KU Leuven
"""Transformations that rewrite DeepProbLog programs before grounding."""

from collections.abc import Iterable

from deeplog import flatten_symbol
from deeplog import get_term_variables
from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog import create_rule
from deeplog.grounding.prolog import is_fact

from .parser import create_labeled_fact
from .parser import get_atom
from .parser import get_label


def remove_labeled_rules(program: Iterable[RuleType]) -> Iterable[RuleType]:
    """Remove labeled rules from a program by introducing a new probabilistic fact and putting it in the body.

    The aux fact is keyed on the head's arguments plus any variables that
    appear in the label but not in the head (e.g. ``Idx`` in
    ``classifier(Img, Idx) :: class(Img, Class) :- nth0(Idx, ..., Class)``).
    Without those extra variables, the aux fact would unify with the head
    fine, but the label term would still carry unbound variables, and the
    resulting label probability lookup would be ambiguous.
    """
    nr_aux_facts = 0
    for rule in program:
        head = rule[1]
        label = get_label(head) if not is_fact(rule) else None
        if label is not None:
            head = get_atom(head)
            seen: set = set()
            for v in get_term_variables(head):
                seen.add(v)
            extra_label_vars: list = []
            for v in get_term_variables(label):
                if v not in seen:
                    seen.add(v)
                    extra_label_vars.append(v)
            aux_symbol = (f"aux{nr_aux_facts}", *head[1:], *extra_label_vars)
            nr_aux_facts += 1
            aux_fact = create_labeled_fact(aux_symbol, label)
            new_rule = create_rule(
                [head], list(flatten_symbol(rule[2], ",")) + [aux_symbol]
            )
            yield aux_fact
            yield new_rule
        else:
            yield rule

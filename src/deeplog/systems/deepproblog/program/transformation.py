#  Copyright (c) 2024-2026. KU Leuven
"""Transformations that rewrite DeepProbLog programs before compilation."""

from collections.abc import Iterable

from deeplog.symbol import flatten_symbol
from deeplog.symbol import get_term_variables

from .program import RuleType
from .program import create_categorical_label
from .program import create_fact
from .program import create_rule
from .program import get_ad_branches
from .program import get_atom
from .program import get_label
from .program import is_annotated_disjunction
from .program import is_fact


def expand_annotated_disjunctions(program: Iterable[RuleType]) -> Iterable[RuleType]:
    """Split each AD-fact ``p1::a1; ...; pn::an.`` into n regular facts.

    Each emitted fact carries a categorical-wrapped label so engines can
    recognise the branches as belonging to the same categorical variable
    and (for engines that enumerate proofs) skip proofs that commit to two
    different branches at once.

    Cat-ids are assigned per-call, monotonically. Non-AD rules pass through
    unchanged.
    """
    cat_id = 0
    for rule in program:
        if is_annotated_disjunction(rule):
            for value_idx, branch in enumerate(get_ad_branches(rule)):
                inner_label = get_label(branch)
                assert (
                    inner_label is not None
                )  # is_annotated_disjunction guarantees this
                yield create_fact(
                    get_atom(branch),
                    create_categorical_label(inner_label, cat_id, value_idx),
                )
            cat_id += 1
        else:
            yield rule


def remove_labeled_rules(program: Iterable[RuleType]) -> Iterable[RuleType]:
    """Remove labeled rules from a program by introducing an new probabilistic fact and putting it in the body.

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
            aux_fact = create_fact(aux_symbol, label)
            new_rule = create_rule(
                [head], list(flatten_symbol(rule[2], ",")) + [aux_symbol]
            )
            yield aux_fact
            yield new_rule
        else:
            yield rule

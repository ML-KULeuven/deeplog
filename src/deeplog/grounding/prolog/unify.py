#  Copyright (c) 2024-2026. KU Leuven
"""Symbolic unification for the Prolog grounder."""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping

from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import calculate_mgu
from deeplog.symbol import get_term_variables


def unify(term1: Symbol, term2: Symbol) -> tuple[Symbol, dict[Symbol, Symbol]] | None:
    """
    Returns the unification of the terms if it exists, returns None otherwise.
    """
    substitution = calculate_mgu(term1, term2)
    if substitution is None:
        return None
    return apply_substitution(term1, substitution), substitution


def replace_with_fresh_variables(
    term: Symbol, fresh_variable_function: Callable[[], Symbol]
) -> tuple[Symbol, dict[Symbol, Symbol]]:
    """

    :param term: The term in which to replace all variables with new variables.
    :param fresh_variable_function: A function that takes no arguments, but returns a new unique variable every time
        it is called.
    :return: The term with new variables, and the substitution used to get from the original term to the new term.
    """
    variable_dict = defaultdict(fresh_variable_function)
    fresh_variables = {var: variable_dict[var] for var in get_term_variables(term)}
    return apply_substitution(term, fresh_variables), fresh_variables


def chain_substitution(
    first: Mapping[Symbol, Symbol], second: Mapping[Symbol, Symbol]
) -> Mapping[Symbol, Symbol]:
    """Return the substitution equivalent to applying ``first`` and then ``second``.

    For keys in ``domain(first)``, the chained image ``second(first(k))`` is
    used. Keys only in ``second`` are carried over unchanged. If a key
    appears in both, the chained image wins -- ``second[k]`` is shadowed
    because ``first`` has already replaced ``k`` by the time ``second``
    would apply.
    """
    result = {k: apply_substitution(v, second) for k, v in first.items()}
    for k, v in second.items():
        result.setdefault(k, v)
    return result

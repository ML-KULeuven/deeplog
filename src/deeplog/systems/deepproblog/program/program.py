"""
A module providing the Program, Rule and Query classes.
"""

#  Copyright (c) 2024-2026. KU Leuven

from collections.abc import Iterable
from typing import TypeGuard
from typing import cast

from deeplog.symbol import FalseSymbol
from deeplog.symbol import Symbol
from deeplog.symbol import TrueSymbol
from deeplog.symbol import flatten_symbol
from deeplog.symbol import parse_symbol
from deeplog.symbol import to_symbol
from deeplog.util import bracket_aware_split
from deeplog.util import foldr


CATEGORICAL_LABEL_FUNCTOR = "@cat"


type RuleType = tuple[str, Symbol, Symbol]
type Program = tuple[RuleType, ...]


def create_rule(
    head_atoms: list[str] | list[Symbol],
    body_atoms: list[str] | list[Symbol],
    functor: str = ":-",
) -> RuleType:
    """
    A helper function for creating rule symbols.
    :param head_atoms: A list of Symbols h1, ..., hn
    :param body_atoms: A list of Symbols b1, ..., bm
    :param functor: The functor of the rule, ':-' by default.
    :return: a Rule h1 ; ... ; hn 'functor' b1, ..., bm.
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


def is_rule(symbol: Symbol) -> TypeGuard[RuleType]:
    """
    Returns true if the given symbol is of arity 3 and the functor is ':-'
    """
    return len(symbol) == 3 and symbol[0] == ":-"


def create_query(body_atoms: list[str] | list[Symbol]) -> RuleType:
    """
    A helper function for creating query symbols.
    :param body_atoms: A list of Symbols b1, ..., bm
    :return: a query Symbol false ?- b1, ..., bm.
    """
    return create_rule([], body_atoms, "?-")


def is_query(symbol: Symbol) -> TypeGuard[RuleType]:
    """
    Returns true if the given symbol is of arity, the functor is '?-' and the head is FalseSymbol.
    """
    return len(symbol) == 3 and symbol[0] == "?-" and symbol[1] == FalseSymbol


def create_constraint(body_atoms: list[str] | list[Symbol]) -> RuleType:
    """
    A helper function for creating constraint symbols.
    :param body_atoms: A list of Symbols b1, ..., bm
    :return: a constraint Symbol false :- b1, ..., bm.
    """
    return create_rule([], body_atoms)


def is_constraint(symbol: Symbol) -> TypeGuard[RuleType]:
    """
    Returns true if the given symbol is a rule and the head is FalseSymbol.
    """
    return is_rule(symbol) and symbol[1] == FalseSymbol


def create_labeled_atom(atom: Symbol | str, label: Symbol | str | None) -> Symbol:
    """
    A helper function for creating labeled atom symbols.
    :param atom: A symbol 'a'
    :param label: An optional label l
    :return: If label is None, it returns 'a', otherwise it returns a symbol l :: a.
    """
    atom = to_symbol(atom)
    if label is None:
        return atom
    return "::", to_symbol(label), atom


def get_label(atom: Symbol) -> Symbol | None:
    """
    Returns the label if the symbol is a labeled atom as constructed by the LabeledAtom function,
    returns None otherwise.
    """
    if len(atom) == 3 and atom[0] == "::":
        return atom[1]
    return None


def get_atom(atom: Symbol) -> Symbol:
    """
    Returns the atom if the symbol is a labeled atom as constructed by the LabeledAtom function,
    returns the symbol itself otherwise.
    """
    if len(atom) == 3 and atom[0] == "::":
        return atom[2]
    return atom


def create_categorical_label(
    inner_label: Symbol | str, cat_id: int, value_idx: int
) -> Symbol:
    """Wrap ``inner_label`` so engines can recognise it as a categorical branch.

    The resulting label has the form
    ``("@cat", inner_label, ("@cat_id_<n>",), ("<value_idx>",))``.
    The ``@cat_id_`` prefix on the cat-id avoids collisions with user-defined
    predicates on the Prolog side, where the wrapped label round-trips as
    ``'@cat'(InnerLabel, '@cat_id_<n>', <value_idx>)``.
    """
    return (
        CATEGORICAL_LABEL_FUNCTOR,
        to_symbol(inner_label),
        (f"@cat_id_{cat_id}",),
        (str(value_idx),),
    )


def is_categorical_label(label: Symbol) -> bool:
    """Return true if ``label`` was produced by :func:`create_categorical_label`."""
    return (
        isinstance(label, tuple)
        and len(label) == 4
        and label[0] == CATEGORICAL_LABEL_FUNCTOR
    )


def unwrap_categorical_label(label: Symbol) -> tuple[Symbol, str, int]:
    """Return ``(inner_label, cat_id, value_idx)`` for a categorical-wrapped label."""
    assert is_categorical_label(label), f"not a categorical label: {label}"
    inner_label = cast(Symbol, label[1])
    cat_id_sym = cast(Symbol, label[2])
    value_idx_sym = cast(Symbol, label[3])
    return inner_label, cast(str, cat_id_sym[0]), int(cast(str, value_idx_sym[0]))


def is_annotated_disjunction(rule: Symbol) -> bool:
    """Return true if ``rule`` is a probabilistic AD-fact ``p1::a1; ...; pn::an.``.

    The parser already splits these on ``;`` into a fact whose head is a
    ``;``-tree of labeled atoms with body ``true``. An AD must have at least
    two labeled branches; a regular labeled fact ``p::a.`` is not an AD.
    """
    if not is_fact(rule):
        return False
    head = rule[1]
    if not (isinstance(head, tuple) and len(head) == 3 and head[0] == ";"):
        return False
    branches = list(flatten_symbol(head, ";"))
    if len(branches) < 2:
        return False
    return all(get_label(b) is not None for b in branches)


def get_ad_branches(rule: RuleType) -> list[Symbol]:
    """Return the labeled-atom branches of an AD-fact (in source order)."""
    return list(flatten_symbol(rule[1], ";"))


def create_fact(atom: Symbol | str, label: Symbol | str | None) -> RuleType:
    """
    A helper function for creating fact symbols.
    :param atom: A symbol 'a'.
    :param label: An optional label l.
    :return: Returns a rule a :: l :- true.
    """
    return create_rule([create_labeled_atom(atom, label)], [])


def is_fact(symbol: Symbol) -> TypeGuard[RuleType]:
    """
    Returns true if the given symbol is a rule and the body is empty (i.e. equal to TrueSymbol).
    """
    return is_rule(symbol) and symbol[2] == TrueSymbol


def get_fact_atom(fact: RuleType) -> Symbol:
    """
    Returns the atom of the fact.
    """
    return get_atom(fact[1])


def get_fact_label(fact: RuleType) -> Symbol | None:
    """
    Returns the label of the fact.
    """
    return get_label(fact[1])


def str_to_rules(code: str) -> Iterable[RuleType]:
    """
    A simple parser function that returns an iterable over all rules in a string of the form
    '''
    a :- b1, b2.
    l :: f.
    ?- q1,q2.
    :- c1,c2.
    ...
    '''
    """
    for line in code.split("\n"):
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        yield str_to_rule(line)


def str_to_atoms(atoms_str: str, split_str) -> Iterable[Symbol]:
    """
    :param atoms_str: A string of atoms, separated by a fixed string.
    :param split_str: The string to split on.
    :return: An iterable over all atoms in the given string.
    """
    for atom_str in bracket_aware_split(atoms_str, split_str):
        atom_str = atom_str.strip()
        if len(atom_str) == 0:
            continue

        label = None
        if "::" in atom_str:
            label_str, atom_str = atom_str.split("::")
            label = parse_symbol(label_str.strip())
        atom = parse_symbol(atom_str.strip())
        yield create_labeled_atom(atom, label)


def str_to_rule(line: str) -> RuleType:
    """

    :param line: A string of the form label :: h1, ..., hn :- b1, ..., bm.
    :return: A parsed symbol of the given line.
    """
    assert line[-1] == "."
    line = line[:-1]
    functor = "?-" if "?-" in line else ":-"
    body_str = ""
    head_str = line
    if functor in line:
        head_str, body_str = line.split(functor)
    head = list(str_to_atoms(head_str, ";"))
    body = list(str_to_atoms(body_str, ","))

    return create_rule(head, body, functor)

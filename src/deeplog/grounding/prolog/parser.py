#  Copyright (c) 2024-2026. KU Leuven
"""Parse plain-Prolog source text into :mod:`~deeplog.grounding.prolog.program` clauses.

The head / body / functor splitting is generic and knows only plain Prolog. An
extension such as DeepProbLog layers its own surface syntax (``p :: a`` labels,
annotated disjunctions) on top by passing its own ``atom_parser`` to
:func:`parse_rule` / :func:`parse_atoms` -- see
``deeplog.systems.deepproblog.parser``.
"""

from collections.abc import Callable
from collections.abc import Iterable

from deeplog.symbol import Symbol
from deeplog.symbol import parse_symbol
from deeplog.util import bracket_aware_split

from .program import RuleType
from .program import create_rule


def parse_atoms(
    atoms_str: str,
    split_str: str,
    atom_parser: Callable[[str], Symbol] = parse_symbol,
) -> Iterable[Symbol]:
    """Parse ``atoms_str`` split on ``split_str`` with ``atom_parser`` per atom.

    ``atom_parser`` defaults to the plain term parser; an extension passes its
    own (e.g. one that peels a ``p ::`` label prefix) to reuse this splitting.
    """
    for atom_str in bracket_aware_split(atoms_str, split_str):
        atom_str = atom_str.strip()
        if len(atom_str) == 0:
            continue
        yield atom_parser(atom_str)


def parse_rule(
    line: str, atom_parser: Callable[[str], Symbol] = parse_symbol
) -> RuleType:
    """Parse one clause line ``h1 ; ... <functor> b1 , ...`` ending in ``.``.

    The head/body/functor splitting is generic; ``atom_parser`` decides how each
    atom is read, so extensions reuse this for their own surface syntax.
    """
    assert line[-1] == "."
    line = line[:-1]
    functor = "?-" if "?-" in line else ":-"
    head_str = line
    body_str = ""
    if functor in line:
        head_str, body_str = line.split(functor)
    head = list(parse_atoms(head_str, ";", atom_parser))
    body = list(parse_atoms(body_str, ",", atom_parser))
    return create_rule(head, body, functor)


def str_to_rule(line: str) -> RuleType:
    """Parse a single plain-Prolog clause line."""
    return parse_rule(line)


def iter_clauses(code: str) -> Iterable[str]:
    """Split source text into ``.``-terminated clause strings.

    A period ends a clause only at bracket depth zero and when followed by
    whitespace or the end of the text, so floats (``0.8::a``) and periods
    inside argument positions never split. Clauses may share a line or span
    several lines (newlines inside a clause are folded to spaces). ``#``
    comment lines are skipped. A trailing fragment without its terminator is
    yielded as-is so the rule parser reports it instead of dropping it
    silently.
    """
    text = "\n".join(
        line for line in code.split("\n") if not line.lstrip().startswith("#")
    )
    depth, start = 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif (
            char == "." and depth == 0 and (i + 1 == len(text) or text[i + 1].isspace())
        ):
            clause = " ".join(text[start : i + 1].split())
            if clause != ".":
                yield clause
            start = i + 1
    rest = text[start:].strip()
    if rest:
        yield " ".join(rest.split())


def str_to_rules(code: str) -> Iterable[RuleType]:
    """Parse a plain-Prolog program; clauses may share a line."""
    for clause in iter_clauses(code):
        yield str_to_rule(clause)

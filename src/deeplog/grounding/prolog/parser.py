#  Copyright (c) 2024-2026. KU Leuven
"""Plain-Prolog source text: parsed into :mod:`~deeplog.grounding.prolog.program` clauses, and rendered from symbols.

The head / body / functor splitting is generic and knows only plain Prolog; an
operator such as ``::`` is read as the ordinary binary term it is.
"""

from collections.abc import Callable
from collections.abc import Iterable
from itertools import count
from typing import cast

from deeplog.symbol import Symbol
from deeplog.symbol import get_term_variables
from deeplog.symbol import parse_symbol
from deeplog.symbol import split_list
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
    atom is read, so extensions reuse this for their own surface syntax. Every
    ``_`` is a variable of its own, as in Prolog: it is named ``_1``, ``_2``,
    ... with names the clause does not already use.
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
    return _name_anonymous_variables(create_rule(head, body, functor))


def _name_anonymous_variables(clause: RuleType) -> RuleType:
    """Name every ``_`` in ``clause`` apart, with names the clause does not use."""
    used = {variable[0] for variable in get_term_variables(clause)}
    names = (f"_{n}" for n in count(1) if f"_{n}" not in used)

    def name(term: Symbol) -> Symbol:
        if term == ("_",):
            return (next(names),)
        return (term[0], *(name(argument) for argument in term[1:]))

    return cast("RuleType", name(clause))


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


def symbol_to_prolog_str(symbol: Symbol) -> str:
    """Convert a Symbol to a valid Prolog term string using strict prefix notation.

    Translates ``cons``/``nil`` chains (deeplog's internal list encoding) to
    Prolog list syntax ``[h1, h2, ... | tail]`` so SWI list builtins like
    ``nth0/3`` and ``member/2`` operate on them.
    """
    list_str = _try_render_prolog_list(symbol)
    if list_str is not None:
        return list_str
    if symbol == ("nil",):
        return "[]"
    functor = symbol[0]
    if len(symbol) == 1:
        return _quote_prolog_atom(functor)
    args = ",".join(symbol_to_prolog_str(s) for s in symbol[1:])
    return f"{_quote_prolog_atom(functor)}({args})"


def _try_render_prolog_list(symbol: Symbol) -> str | None:
    """Render a ``cons``/``nil`` chain as ``[h1, h2, ... | tail]``; else None."""
    elements, tail = split_list(symbol)
    if not elements:
        return None
    rendered = ",".join(symbol_to_prolog_str(element) for element in elements)
    if tail == ("nil",):
        return f"[{rendered}]"
    return f"[{rendered}|{symbol_to_prolog_str(tail)}]"


def _quote_prolog_atom(s: str) -> str:
    if _is_prolog_bare_token(s):
        return s
    escaped = s.replace("'", "\\'")
    return f"'{escaped}'"


def _is_prolog_bare_token(s: str) -> bool:
    """Return true if ``s`` can be emitted bare (unquoted) into Prolog source.

    Bare-safe: numbers (parsed as int/float), lowercase identifiers
    ``[a-z][a-zA-Z0-9_]*``, and variable-form names (uppercase- or
    underscore-prefixed identifiers) which Prolog should read as
    variables -- quoting them would convert them into atoms.
    """
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    if not all(c.isalnum() or c == "_" for c in s):
        return False
    return s[0].islower() or s[0].isupper() or s[0] == "_"

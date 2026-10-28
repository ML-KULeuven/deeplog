#  Copyright (c) 2024-2026. KU Leuven
"""Plain-Prolog source text: parsed into :mod:`~deeplog.grounding.prolog.program` clauses, and rendered from symbols.

The head / body / functor splitting is generic and knows only plain Prolog; an
operator such as ``::`` is read as the ordinary binary term it is.
"""

import re
from collections.abc import Callable
from collections.abc import Iterable
from itertools import count
from typing import cast

from deeplog.symbol import Symbol
from deeplog.symbol import get_term_variables
from deeplog.symbol import parse_symbol
from deeplog.util import bracket_aware_split

from .program import RuleType
from .program import create_rule


#: The operators :func:`parse_term` reads before arithmetic, in the order it
#: splits a term on them: a term is read around the first one outside brackets.
#: ``is`` is split on only where space surrounds it.
_OPERATORS = (":-", "?-", "::", ";", ",", " is ")

#: Prolog's infix arithmetic operators and their priorities. The lower the
#: priority, the tighter the operator binds; ``^`` groups to the right, the
#: others to the left.
_ARITHMETIC_PRIORITIES = {
    "+": 500,
    "-": 500,
    "/\\": 500,
    "\\/": 500,
    "xor": 500,
    "*": 400,
    "/": 400,
    "//": 400,
    "rem": 400,
    "mod": 400,
    "div": 400,
    "<<": 400,
    ">>": 400,
    "^": 200,
}
_ARITHMETIC_HINT = re.compile(r"[-+*/\\^<>]|\b(?:xor|rem|mod|div)\b")
_ARITHMETIC_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
    | (?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<quoted>'(?:[^'\\]|\\.)*')
    | (?P<name>[^\W\d]\w*)
    | (?P<symbol>[-+*/\\^<>=~:.?@#&$]+)
    | (?P<group>[(\[])
    """,
    re.VERBOSE,
)


def parse_term(text: str) -> Symbol:
    """Read the Prolog term ``text``.

    A term is read around the first of ``:-``, ``?-``, ``::``, ``;``, ``,`` and
    ``is`` outside brackets; ``,`` and ``;`` group to the right. Otherwise, text
    made of operands joined by the arithmetic operators ``+ - * / // mod rem div
    ^ /\\ \\/ xor << >>`` is read with Prolog's priorities and associativity:
    ``Z is (N1 + N2) mod 10`` is ``is(Z, mod(+(N1, N2), 10))``. Any other text
    is read by :func:`~deeplog.symbol.parse_symbol`, with each argument read as
    a Prolog term.
    """
    text = text.strip()
    term = _parse_operator(text)
    if term is None:
        term = _parse_arithmetic(text)
    if term is None:
        term = parse_symbol(text, parse_argument=parse_term)
    return term


def _parse_operator(text: str) -> Symbol | None:
    """Read ``text`` around one of :data:`_OPERATORS`, or return ``None``."""
    for operator in _OPERATORS:
        parts = list(bracket_aware_split(text, operator))
        if len(parts) == 1:
            continue
        functor = operator.strip()
        if functor in (",", ";"):
            term = parse_term(parts[-1])
            for part in reversed(parts[:-1]):
                term = (functor, parse_term(part), term)
            return term
        if len(parts) != 2:
            return None
        left, right = parts
        if not left.strip() or not right.strip():
            continue
        return functor, parse_term(left), parse_term(right)
    return None


def _parse_arithmetic(text: str) -> Symbol | None:
    """Read ``text`` as infix arithmetic, or return ``None`` if it is not.

    ``N1 + N2 * 2`` is ``+(N1, *(N2, 2))``: the operands are read on their own
    and the operators bind by their Prolog priority. Only text that alternates
    operands and arithmetic operators is read, so a lone atom, a signed number
    and ``X =:= Y + 1`` (an operator this does not read) all return ``None``.
    """
    if not _ARITHMETIC_HINT.search(text):
        return None
    tokens = _arithmetic_tokens(text)
    if tokens is None or len(tokens) < 3 or len(tokens) % 2 == 0:
        return None
    operands, operators = tokens[0::2], tokens[1::2]
    if any(kind == "symbol" for kind, _ in operands) or any(
        kind not in ("symbol", "name") or text not in _ARITHMETIC_PRIORITIES
        for kind, text in operators
    ):
        return None
    terms = [
        parse_term(text[1:-1] if text.startswith("(") else text) for _, text in operands
    ]
    tree, _ = _fold_arithmetic(terms, [text for _, text in operators], 0, 1200)
    return tree


def _arithmetic_tokens(text: str) -> list[tuple[str, str]] | None:
    """Split ``text`` into its operands and operators, or ``None`` if it cannot.

    Each token is ``(kind, text)``. A ``term`` is an operand: a number, a quoted
    atom, a bracketed group, or a name or quoted atom with its arguments. A
    ``symbol`` is a run of symbol characters and a ``name`` a bare identifier,
    either of which may be an operator.
    """
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _ARITHMETIC_TOKEN.match(text, position)
        if match is None:
            return None
        kind, start, position = match.lastgroup, match.start(), match.end()
        if kind == "space":
            continue
        if kind == "group" or (
            kind in ("name", "quoted") and text.startswith("(", position)
        ):
            end = _bracket_group_end(
                text, position - 1 if kind == "group" else position
            )
            if end is None:
                return None
            kind, position = "term", end
        elif kind in ("number", "quoted"):
            kind = "term"
        tokens.append((str(kind), text[start:position]))
    return tokens


def _bracket_group_end(text: str, start: int) -> int | None:
    """The index just past the bracket group opening at ``start``, or ``None``."""
    depth = 0
    quoted = False
    position = start
    while position < len(text):
        char = text[position]
        if quoted:
            if char == "\\":
                position += 1
            elif char == "'":
                quoted = False
        elif char == "'":
            quoted = True
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                return position + 1
        position += 1
    return None


def _fold_arithmetic(
    terms: list[Symbol], operators: list[str], index: int, max_priority: int
) -> tuple[Symbol, int]:
    """Fold ``terms[index:]`` while the next operator's priority is at most ``max_priority``.

    ``operators[i]`` joins ``terms[i]`` and ``terms[i + 1]``. Returns the folded
    tree and the index of the first operator left unfolded.
    """
    tree = terms[index]
    while index < len(operators):
        operator = operators[index]
        priority = _ARITHMETIC_PRIORITIES[operator]
        if priority > max_priority:
            break
        right_priority = priority if operator == "^" else priority - 1
        right, index = _fold_arithmetic(terms, operators, index + 1, right_priority)
        tree = (operator, tree, right)
    return tree, index


def parse_atoms(
    atoms_str: str,
    split_str: str,
    atom_parser: Callable[[str], Symbol] = parse_term,
) -> Iterable[Symbol]:
    """Parse ``atoms_str`` split on ``split_str`` with ``atom_parser`` per atom.

    ``atom_parser`` defaults to :func:`parse_term`; an extension passes its
    own (e.g. one that peels a ``p ::`` label prefix) to reuse this splitting.
    """
    for atom_str in bracket_aware_split(atoms_str, split_str):
        atom_str = atom_str.strip()
        if len(atom_str) == 0:
            continue
        yield atom_parser(atom_str)


def parse_rule(
    line: str, atom_parser: Callable[[str], Symbol] = parse_term
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


def split_list(term: Symbol) -> tuple[list[Symbol], Symbol]:
    """Return the elements of the list ``term`` and the tail they end in.

    A list is a ``cons``/``nil`` chain, so ``[a, b]`` is ``([a, b], nil)``,
    ``[a | T]`` is ``([a], T)``, and a term that is not a list is
    ``([], term)``.
    """
    elements: list[Symbol] = []
    while len(term) == 3 and term[0] == "cons":
        elements.append(term[1])
        term = term[2]
    return elements, term


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

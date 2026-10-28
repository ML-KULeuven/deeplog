#  Copyright (c) 2024-2026. KU Leuven
"""
A module that provides the Symbol type.
"""

from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from itertools import chain
from typing import Any
from typing import TypeGuard
from typing import overload

from .util import bracket_aware_split


# TODO(post-v3.0.0): reconsider the Symbol representation. The flattened
# `tuple[str, *tuple[Symbol, ...]]` shape is cheap and hashable but defeats
# nominal typing (Variable/Atom/Compound) and forces pyright ignores in the
# runtime guards. Candidates: a frozen-dataclass hierarchy, NamedTuple, or
# typed factory functions + TypeGuards over the existing tuple shape.
#: A term as a nested tuple — a functor followed by its arguments, each a symbol
#: in turn. ``("digit", ("i1",), ("3",))`` is ``digit(i1,3)``, and a bare
#: ``("a",)`` is the constant ``a``.
type Symbol = tuple[str, *tuple["Symbol", ...]]

TrueSymbol = ("true",)
FalseSymbol = ("false",)


def is_symbol(symbol: Any) -> TypeGuard[Symbol]:
    """Return whether ``symbol`` is a well-formed symbolic tuple."""
    # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType] —
    # this is runtime validation of arbitrary input; element types are unknown
    # by design until this function returns True.
    return (
        isinstance(symbol, tuple)
        and len(symbol) > 0  # pyright: ignore[reportUnknownArgumentType]
        and isinstance(symbol[0], str)
        and all(is_symbol(a) for a in symbol[1:])  # pyright: ignore[reportUnknownVariableType]
    )


def is_variable(symbol: Symbol) -> bool:
    """Return whether ``symbol`` represents a variable."""
    return (
        is_symbol(symbol)
        and len(symbol) == 1
        and (symbol[0][0].isupper() or symbol[0][0] == "_")
    )


def get_predicate(symbol: Symbol) -> tuple[str, int]:
    """Return the predicate functor and arity for the given symbol."""
    return symbol[0], len(symbol) - 1


def get_args(symbol: Symbol) -> tuple[Symbol, ...]:
    """Return the arguments of ``symbol`` (everything after the functor)."""
    return symbol[1:]  # pyright: ignore[reportReturnType]


def is_structure_wrapped(symbol: Symbol) -> bool:
    """Return whether ``symbol`` has the form ``("_", inner, (structure,))``."""
    return len(symbol) == 3 and symbol[0] == "_"


def structure_of(symbol: Symbol) -> str | None:
    """Return the algebraic structure ``symbol`` is labelled with, or ``None``.

    The total read of the ``("_", inner, (structure,))`` tag: a bare symbol
    names a value that lives in no algebra, which is a legitimate state outside
    the formula layer, so absence is reported rather than raised. Counterpart
    of :func:`with_structure`.
    """
    return symbol[2][0] if is_structure_wrapped(symbol) else None


def unwrap_structure(atom: Symbol) -> Symbol:
    """Return the inner symbol of a structure-wrapped atom (``atom[1]``).

    The strict unwrap, used at the circuit boundary where a tag is required by
    construction. Raises ``ValueError`` if ``atom`` is not structure-wrapped;
    use :func:`without_structure` where a bare symbol is acceptable.
    """
    if is_structure_wrapped(atom):
        return atom[1]  # pyright: ignore[reportReturnType]
    raise ValueError(f"Invalid atom: {atom}")


def without_structure(symbol: Symbol) -> Symbol:
    """Return ``symbol`` with any structure tag removed.

    The tolerant counterpart of :func:`unwrap_structure`: a bare symbol is
    returned unchanged. Used when re-minting a name from a labelled one, so the
    tag is re-applied to the *outside* rather than buried in the new symbol.
    """
    return symbol[1] if is_structure_wrapped(symbol) else symbol  # pyright: ignore[reportReturnType]


def with_structure(atom: Symbol, structure: str) -> Symbol:
    """Return ``atom`` wrapped with ``structure``, replacing any existing wrapper."""
    if is_structure_wrapped(atom):
        atom = atom[1]  # pyright: ignore[reportAssignmentType]
    return "_", atom, (structure,)


def retag(new: Symbol, like: Symbol) -> Symbol:
    """Return ``new`` carrying the structure tag of ``like``, bare if it has none.

    What a module applies when it mints an output name for a value it did not
    change the algebra of — an aggregation naming its reduction, an elementwise
    operator naming its column.
    """
    structure = structure_of(like)
    return with_structure(new, structure) if structure is not None else new


def strip_literal_structure(symbol: Symbol, structure: str) -> Symbol:
    """Validate ``symbol``'s structure tag matches ``structure`` and drop the wrapper.

    For unwrapped symbols, returns ``symbol`` unchanged.
    """
    if is_structure_wrapped(symbol):
        embedded_structure = symbol[2][0]
        if embedded_structure != structure:
            raise ValueError(
                f"Leaf node structure mismatch: expected {structure}, "
                f"got {embedded_structure}"
            )
        return "_", symbol[1]  # pyright: ignore[reportReturnType]
    return symbol


def get_term_variables(symbol: Symbol) -> Iterable[Symbol]:
    """
    Returns an iterable of all variables that appear in Symbol. These are not unique.
    """
    if is_variable(symbol):
        yield symbol
    else:
        yield from chain.from_iterable(get_term_variables(arg) for arg in symbol[1:])


def free_variables(symbol: Symbol) -> tuple[Symbol, ...]:
    """The variables free in ``symbol``, each once, in the order they first occur.

    A ``binders(V1, ..., Vn)`` argument binds its variables in the arguments
    beside it, as an aggregation's name does: ``sum(binders(X), p(X, Y))`` has
    ``Y`` free and not ``X``.
    """
    found: dict[Symbol, None] = {}

    def visit(term: Symbol, bound: frozenset[Symbol]) -> None:
        if is_variable(term):
            if term not in bound:
                found.setdefault(term, None)
            return
        arguments = term[1:]
        binding = [a for a in arguments if a[0] == "binders"]
        inner = bound.union(*(set(get_term_variables(b)) for b in binding))
        for argument in arguments:
            if argument[0] != "binders":
                visit(argument, inner)  # pyright: ignore[reportArgumentType]

    visit(symbol, frozenset())
    return tuple(found)


#: The functors :func:`symbol_to_pretty_string` writes between their operands.
infix_functors = {":-", "?-", "::", ";", ",", "is", "_"}
associative_infix_functors = {",", ";"}  # chain to the right, as Prolog's xfy


def _closing_bracket(text: str, opening: int) -> int | None:
    """The position of the bracket closing the one at ``opening``, or ``None``."""
    depth = 0
    for position in range(opening, len(text)):
        if text[position] in "([":
            depth += 1
        elif text[position] in ")]":
            depth -= 1
            if depth == 0:
                return position
    return None


def parse_symbol(
    symbol_str: str, parse_argument: Callable[[str], Symbol] | None = None
) -> Symbol:
    """Read the symbol ``symbol_str``.

    A symbol is a constant or variable ``a``, a compound term ``f(a, b)``, a list
    ``[a, b | T]``, or a structure-tagged symbol ``a _ s``, and brackets around
    a symbol group it. Text without brackets that is none of these, such as
    ``X is Y``, is one constant.

    Args:
        symbol_str: The text to read.
        parse_argument: Reads each argument, list element, side of ``_`` and
            bracketed symbol; :func:`parse_symbol` itself by default.

    Raises:
        ValueError: If a term's arguments are not closed, text follows the
            bracket closing them, or brackets enclose nothing.
    """
    parse_argument = parse_argument or parse_symbol
    symbol_str = symbol_str.strip()
    if symbol_str == "_":
        return ("_",)
    if symbol_str.startswith("[") and symbol_str.endswith("]"):
        return _parse_list(symbol_str, parse_argument)
    parts = list(bracket_aware_split(symbol_str, " _ "))
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return "_", parse_argument(parts[0]), parse_argument(parts[1])
    first_bracket = symbol_str.find("(")
    if first_bracket == -1:
        return (symbol_str,)
    final_bracket = _closing_bracket(symbol_str, first_bracket)
    if final_bracket is None:
        raise ValueError(f"No closing bracket found while parsing {symbol_str}.")
    if final_bracket != len(symbol_str) - 1:
        raise ValueError(
            f"{symbol_str!r} continues after the bracket closing its arguments. "
            "A structure tag `_` is written with spaces around it."
        )
    functor = symbol_str[:first_bracket]
    args_str = symbol_str[first_bracket + 1 : final_bracket]
    if not functor:
        if not args_str.strip():
            raise ValueError(f"{symbol_str!r} brackets nothing.")
        return parse_argument(args_str)
    args = tuple(bracket_aware_split(args_str, ","))
    return functor, *(parse_argument(a) for a in args)


@overload
def to_symbol(arg: Symbol) -> Symbol: ...
@overload
def to_symbol(arg: str) -> Symbol: ...
@overload
def to_symbol(arg: list[str] | list[Symbol]) -> list[Symbol]: ...
def to_symbol(arg: object) -> Symbol | list[Symbol]:
    """Turn the given argument into a symbol (or list of symbols)."""
    if isinstance(arg, str):
        return parse_symbol(arg)
    if isinstance(arg, list):
        return [to_symbol(x) for x in arg]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    if is_symbol(arg):
        return arg
    raise TypeError(f"Cannot convert {type(arg).__name__} to Symbol")


def symbol_to_str(symbol: Symbol) -> str:
    """Format ``symbol`` as the text :func:`parse_symbol` reads back.

    A compound term is ``f(a,b)``, whatever its functor, so ``is(X,Y)`` for
    ``X is Y``, and a structure tag is ``a _ s``. Brackets group an operand
    that would read otherwise: a side of a tag that is itself tagged, as in
    ``(a _ s) _ t``, and an argument whose functor holds a comma, as in
    ``f((,(a,b)))``. A functor holding a bracket does not read back.

    Use :func:`symbol_to_pretty_string` for human-readable output.
    """
    functor = symbol[0]
    if len(symbol) == 1:
        return functor
    if functor == "_" and len(symbol) == 3:
        return f"{_tagged_str(symbol[1])} _ {_tagged_str(symbol[2])}"
    args = ",".join(_argument_str(s) for s in symbol[1:])
    return f"{functor}({args})"


def _tagged_str(symbol: Symbol) -> str:
    """``symbol`` written as a side of a structure tag, bracketed if it is one."""
    text = symbol_to_str(symbol)
    return f"({text})" if len(symbol) == 3 and symbol[0] == "_" else text


def _argument_str(symbol: Symbol) -> str:
    """``symbol`` written as an argument, bracketed if its functor holds a comma."""
    text = symbol_to_str(symbol)
    return f"({text})" if "," in symbol[0] else text


def symbol_to_pretty_string(symbol: Symbol) -> str:
    """Format ``symbol`` for a human reader, infixing the functors that read better.

    Not the inverse of :func:`parse_symbol`; :func:`symbol_to_str` is the form
    that parses back.
    """
    return _pretty_string(symbol, parenthesize=False)


def _pretty_string(symbol: Symbol, *, parenthesize: bool) -> str:
    """Format ``symbol``, bracketing a flattened infix chain when it is nested.

    A chain needs no brackets as the whole expression (``a :- b``) and does need
    them as a subexpression (``a , (b ; c)``).
    """
    if len(symbol) == 1:
        return symbol[0]
    if (
        len(symbol) == 3
        and (symbol[0] == ":-" or symbol[0] == "?-")
        and symbol[1] == FalseSymbol
    ):
        return f"{symbol[0]} {_pretty_string(symbol[2], parenthesize=False)}"
    if len(symbol) == 3 and symbol[0] in infix_functors:
        # :- infix without ( ) around head and body
        if symbol[0] in (":-", "::", "is"):
            lhs = _pretty_string(symbol[1], parenthesize=False)
            rhs = _pretty_string(symbol[2], parenthesize=False)
            return f"{lhs} {symbol[0]} {rhs}"
        # associative infix functors use flattening
        if symbol[0] in associative_infix_functors:
            result = f" {symbol[0]} ".join(
                _pretty_string(s, parenthesize=True) for s in flatten_symbol(symbol)
            )
            return f"({result})" if parenthesize else result
        # otherwise infix with ( ) around arguments
        lhs = _pretty_string(symbol[1], parenthesize=True)
        rhs = _pretty_string(symbol[2], parenthesize=True)
        return f"({lhs} {symbol[0]} {rhs})"
    args = ",".join(_pretty_string(s, parenthesize=True) for s in symbol[1:])
    return f"{symbol[0]}({args})"


def flatten_symbol(symbol: Symbol, functor: str | None = None) -> Iterable[Symbol]:
    """
    Flattens a tree of functor/2 symbols into an iterable.
    """
    if functor is None:
        functor = symbol[0]
    if len(symbol) == 3 and symbol[0] == functor:
        yield from flatten_symbol(symbol[1], functor)
        yield from flatten_symbol(symbol[2], functor)
    else:
        yield symbol


def apply_substitution(term: Symbol, substitution: Mapping[Symbol, Symbol]) -> Symbol:
    """
    Apply a substitution mapping to a symbolic term.

    :param term: The term to perform the substitution on.
    :param substitution: A dictionary {..., vi -> ti, ...}.
    :return: A new term where each instance of vi in the original term is replaced with ti.
    """
    if substitution.get(term) is not None:
        substituted_term = substitution[term]
        return apply_substitution(substituted_term, substitution)
    substituted_args = list(apply_substitution(arg, substitution) for arg in term[1:])
    return term[0], *substituted_args


def _replace_all_occurrences(
    queue: deque[tuple[Symbol, Symbol]], substitution: dict[Symbol, Symbol]
) -> deque[tuple[Symbol, Symbol]]:
    return deque(
        (apply_substitution(lhs, substitution), apply_substitution(rhs, substitution))
        for lhs, rhs in queue
    )


def calculate_mgu(term1: Symbol, term2: Symbol) -> dict[Symbol, Symbol] | None:
    """Return the most general unifier of ``term1`` and ``term2``, or ``None``.

    The unifier is the substitution ``s`` with
    ``apply_substitution(term1, s) == apply_substitution(term2, s)`` that every
    other such substitution refines. ``None`` means the terms do not unify.
    Variables are identified by name, so every ``_`` is the same variable.

    There is no occurs check. Unifying a variable with a term containing it is
    not supported: it may raise :class:`RecursionError`, or return a substitution
    :func:`apply_substitution` cannot apply.
    """
    queue: deque[tuple[Symbol, Symbol]] = deque([(term1, term2)])
    substitution: dict[Symbol, Symbol] = {}
    while queue:
        lhs, rhs = queue.popleft()
        if is_variable(rhs) and not is_variable(lhs):
            queue.appendleft((rhs, lhs))
        elif is_variable(lhs):
            if lhs == rhs:
                continue
            new_sub = {lhs: rhs}
            substitution = {
                k: apply_substitution(v, new_sub) for k, v in substitution.items()
            }
            substitution.update(new_sub)
            queue = _replace_all_occurrences(queue, new_sub)
        else:
            if get_predicate(lhs) != get_predicate(rhs):
                return None
            queue.extend(zip(lhs[1:], rhs[1:], strict=True))

    return substitution


def _parse_list(list_str: str, parse_argument: Callable[[str], Symbol]) -> Symbol:
    assert list_str[0] == "[" and list_str[-1] == "]"
    inner = list_str[1:-1].strip()
    head_part, tail_part = _split_list_head_tail(inner)

    elements: list[Symbol] = []
    if head_part.strip():
        for element_str in bracket_aware_split(head_part, ","):
            element_str = element_str.strip()
            if not element_str:
                continue
            elements.append(parse_argument(element_str))

    tail_symbol = parse_argument(tail_part.strip()) if tail_part else ("nil",)
    current = tail_symbol
    for element in reversed(elements):
        current = ("cons", element, current)
    return current


def _split_list_head_tail(list_contents: str) -> tuple[str, str | None]:
    depth = 0
    for idx, char in enumerate(list_contents):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "|" and depth == 0:
            return list_contents[:idx], list_contents[idx + 1 :]
    return list_contents, None

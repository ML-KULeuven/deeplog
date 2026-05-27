#  Copyright (c) 2024-2026. KU Leuven

import pytest

from deeplog.symbol import parse_symbol
from deeplog.symbol import symbol_to_pretty_string
from deeplog.symbol import symbol_to_str
from deeplog.systems.deepproblog.program import create_fact
from deeplog.systems.deepproblog.program import create_query
from deeplog.systems.deepproblog.program import create_rule


symbol_examples = [
    ("true", ("true",)),
    ("parentOf(an,bob)", ("parentOf", ("an",), ("bob",))),
]


@pytest.mark.parametrize(["symbol_str", "symbol"], symbol_examples)
def test_symbol(symbol_str, symbol):
    assert parse_symbol(symbol_str) == symbol
    assert symbol_to_str(symbol) == symbol_str


def test_space_in_symbol_str():
    symbol1 = parse_symbol("input(a,b,c)")
    symbol2 = parse_symbol("input(a, b, c)")
    gt_symbol = ("input", ("a",), ("b",), ("c",))
    assert gt_symbol == symbol1 == symbol2


def test_no_closing_bracket():
    with pytest.raises(ValueError):
        parse_symbol("parentOf(an,bob")


def test_list_literal_symbol():
    assert parse_symbol("[a,b,c]") == (
        "cons",
        ("a",),
        ("cons", ("b",), ("cons", ("c",), ("nil",))),
    )


def test_list_literal_with_tail():
    assert parse_symbol("[a|T]") == ("cons", ("a",), ("T",))


def test_nested_list_symbol():
    assert parse_symbol("[a,[b,c],d|T]") == (
        "cons",
        ("a",),
        (
            "cons",
            ("cons", ("b",), ("cons", ("c",), ("nil",))),
            ("cons", ("d",), ("T",)),
        ),
    )


def test_empty_list():
    assert parse_symbol("[]") == ("nil",)


def test_infix_functor_parsing():
    assert parse_symbol("a _ b") == ("_", ("a",), ("b",))
    #    assert parse_symbol("a_b") == ("_", ("a",), ("b",))
    assert parse_symbol("aux0") == ("aux0",)
    assert parse_symbol("X is Y") == ("is", ("X",), ("Y",))
    assert parse_symbol("a , b , c") == (",", ("a",), (",", ("b",), ("c",)))


rule_examples = [
    (
        create_rule(
            [("addition", ("X",), ("Y",), ("Z",))],
            [
                ("digit", ("X",), ("N1",)),
                ("digit", ("Y",), ("N2",)),
                ("is", ("Z",), ("N1+N2",)),
            ],
        ),
        "addition(X,Y,Z) :- digit(X,N1) , digit(Y,N2) , Z is N1+N2",
    ),
    (
        create_fact(("digit", ("X",), ("Y",)), ("nn", ("X",), ("Y",))),
        "nn(X,Y) :: digit(X,Y) :- true",
    ),
    (create_query([("addition", ("a",), ("b",), ("c",))]), "?- addition(a,b,c)"),
]

pretty_string_examples = [
    ((",", (";", "a", "b"), ("not", "a")), "(a ; b) , not(a)"),
    ((";", "a", (",", "b", ("not", "a"))), "a ; (b , not(a))"),
    ((",", "a", (",", "b", "c")), ("a , b , c")),
    ((",", "a", (",", (";", "b", (";", "d", "e")), "c")), ("a , (b ; d ; e) , c")),
] + rule_examples


@pytest.mark.parametrize(["symbol", "pretty_str"], pretty_string_examples)
def test_symbol_to_pretty_string(symbol, pretty_str):
    assert symbol_to_pretty_string(symbol) == pretty_str


a, b, c = ((x,) for x in "abc")


# --- Structure-wrapping helpers ---


def test_get_structure_returns_structure_tuple():
    from deeplog.symbol import get_structure

    atom = ("_", ("v1",), ("boolean",))
    assert get_structure(atom) == ("boolean",)


def test_get_structure_rejects_unwrapped_atom():
    from deeplog.symbol import get_structure

    with pytest.raises(ValueError):
        get_structure(("v1",))


def test_get_structure_rejects_non_underscore_head():
    from deeplog.symbol import get_structure

    with pytest.raises(ValueError):
        get_structure(("=", ("v1",), ("true",)))


def test_with_structure_wraps_unwrapped_atom():
    from deeplog.symbol import with_structure

    assert with_structure(("v1",), "boolean") == ("_", ("v1",), ("boolean",))


def test_with_structure_overrides_existing_structure():
    from deeplog.symbol import with_structure

    wrapped = ("_", ("v1",), ("boolean",))
    assert with_structure(wrapped, "real") == ("_", ("v1",), ("real",))


def test_strip_literal_structure_unwraps_matching_structure():
    from deeplog.symbol import strip_literal_structure

    wrapped = ("_", ("v1",), ("boolean",))
    assert strip_literal_structure(wrapped, "boolean") == ("_", ("v1",))


def test_strip_literal_structure_passes_through_unwrapped_symbol():
    from deeplog.symbol import strip_literal_structure

    assert strip_literal_structure(("v1",), "boolean") == ("v1",)


def test_strip_literal_structure_raises_on_mismatch():
    from deeplog.symbol import strip_literal_structure

    wrapped = ("_", ("v1",), ("boolean",))
    with pytest.raises(ValueError):
        strip_literal_structure(wrapped, "real")

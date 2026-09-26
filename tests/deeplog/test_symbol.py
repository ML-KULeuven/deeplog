#  Copyright (c) 2024-2026. KU Leuven

import pytest

from deeplog.grounding.prolog import create_fact
from deeplog.grounding.prolog import create_query
from deeplog.grounding.prolog import create_rule
from deeplog.symbol import parse_symbol
from deeplog.symbol import split_list
from deeplog.symbol import symbol_to_pretty_string
from deeplog.symbol import symbol_to_str


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


@pytest.mark.parametrize(
    ("symbol_str", "elements", "tail"),
    [
        ("[a,b]", ["a", "b"], "[]"),
        ("[a,[b,c],d|T]", ["a", "[b,c]", "d"], "T"),
        ("[]", [], "[]"),
        ("f(a)", [], "f(a)"),
    ],
    ids=["list", "nested list with a tail", "empty list", "not a list"],
)
def test_split_list(symbol_str, elements, tail):
    """A list splits into its elements and the tail they end in."""
    assert split_list(parse_symbol(symbol_str)) == (
        [parse_symbol(element) for element in elements],
        parse_symbol(tail),
    )


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
        create_fact(("::", ("nn", ("X",), ("Y",)), ("digit", ("X",), ("Y",)))),
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


def test_structure_of_returns_the_structure_name():
    from deeplog.symbol import structure_of

    assert structure_of(("_", ("v1",), ("boolean",))) == "boolean"


def test_structure_of_reports_an_unwrapped_atom_as_unlabelled():
    from deeplog.symbol import structure_of

    assert structure_of(("v1",)) is None


def test_structure_of_reports_a_non_underscore_head_as_unlabelled():
    from deeplog.symbol import structure_of

    assert structure_of(("=", ("v1",), ("true",))) is None


def test_without_structure_is_the_tolerant_unwrap():
    from deeplog.symbol import unwrap_structure
    from deeplog.symbol import without_structure

    assert without_structure(("_", ("v1",), ("boolean",))) == ("v1",)
    assert without_structure(("v1",)) == ("v1",)
    with pytest.raises(ValueError):
        unwrap_structure(("v1",))


def test_retag_carries_the_label_of_the_symbol_it_renames():
    from deeplog.symbol import retag

    labelled = ("_", ("v1",), ("boolean",))
    assert retag(("sum", ("v1",)), labelled) == (
        "_",
        ("sum", ("v1",)),
        ("boolean",),
    )
    # A bare source leaves the new name bare — nothing is invented.
    assert retag(("sum", ("v1",)), ("v1",)) == ("sum", ("v1",))


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

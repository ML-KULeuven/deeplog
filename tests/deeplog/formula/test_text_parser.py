#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch
from lark import LexError
from lark import ParseError

from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import SymTensor
from deeplog import Transformation
from deeplog import UnaryOp
from deeplog import parse_formula
from deeplog import parse_formula_to_module
from deeplog import with_structure


def _weighted_formula():
    burglary = ("Burglary",)
    earthquake = ("Earthquake",)
    true = ("true",)

    b_boolean = Atom(("_", ("=", burglary, true), ("boolean",)))
    e_boolean = Atom(("_", ("=", earthquake, true), ("boolean",)))
    transformed = Transformation("probability", BinaryOp("or", b_boolean, e_boolean))

    b_prob = Atom(("_", ("p", burglary), ("probability",)))
    e_prob = Atom(("_", ("p", earthquake), ("probability",)))
    joint_prob = BinaryOp("times", b_prob, e_prob)

    product = BinaryOp("times", transformed, joint_prob)
    return Aggregation("sum", (burglary, earthquake), (), product)


# --- Canonical round-trip: str(parse_formula(text)) == text ---


def test_parse_leaf():
    text = "=(Burglary,true)_boolean"
    assert str(parse_formula(text)) == text


def test_parse_unary_expression():
    text = "sum(Burglary): not =(Burglary,true)_boolean"
    assert str(parse_formula(text)) == text


def test_parse_general_aggregation():
    text = "product(X): p(X)_probability"
    assert str(parse_formula(text)) == text


def test_parse_aggregation_with_params():
    text = "expect(X; q(X)_probability): p(X)_probability"
    assert str(parse_formula(text)) == text


def test_parse_transformation_only():
    text = "(=(Burglary,true)_boolean)_probability"
    assert str(parse_formula(text)) == text


def test_parse_binary_of_transformations():
    text = "(=(Burglary,true)_boolean)_probability times (=(Earthquake,true)_boolean)_probability"
    assert str(parse_formula(text)) == text


def test_parse_unary_over_transformation():
    text = "sum(Burglary): not (=(Burglary,true)_boolean)_probability"
    assert str(parse_formula(text)) == text


def test_functor_against_its_arguments_is_an_atom():
    """``q(x_bar)`` also parses as a formula, but written against ``q`` it is
    the atom's argument; with a space, ``q`` is an operator over a cast."""
    assert parse_formula("q(x_bar)_boolean") == Atom(
        ("_", ("q", ("x_bar",)), ("boolean",))
    )
    assert parse_formula("q (x_bar)_boolean") == UnaryOp(
        "q", Transformation("boolean", Atom(("_", ("x",), ("bar",))))
    )


def test_an_atoms_arguments_nest():
    assert parse_formula("p(V0,nn(0))_probability") == Atom(
        ("_", ("p", ("V0",), ("nn", ("0",))), ("probability",))
    )
    assert parse_formula("f(g(h(a)),b)_boolean") == Atom(
        ("_", ("f", ("g", ("h", ("a",))), ("b",)), ("boolean",))
    )


def test_a_distribution_names_a_nested_input():
    module = parse_formula_to_module(
        "expectation(V; p(V,nn(0))_probability): =(V,true)_boolean"
    )
    assert module.get_input_shape() == (
        SymTensor([with_structure(("nn", ("0",)), "probability")]),
    )
    assert torch.allclose(module(torch.tensor([[0.3]])), torch.tensor([[0.3]]))


def test_text_nested_thousands_deep_parses():
    casts = ("logprobability" if i % 2 == 0 else "probability" for i in range(2_000))
    text = "(" * 2_000 + "a_probability" + "".join(f")_{cast}" for cast in casts)

    module = parse_formula_to_module(text)

    torch.testing.assert_close(module(torch.tensor([[0.3]])), torch.tensor([[0.3]]))


# --- Normalization: non-canonical input → canonical output ---


def test_parse_model_count_formula():
    text = """
		sum(Burglary, Earthquake):
			=(Burglary,true)_boolean or =(Earthquake,true)_boolean
		"""
    assert (
        str(parse_formula(text))
        == "sum(Burglary, Earthquake): =(Burglary,true)_boolean or =(Earthquake,true)_boolean"
    )


def test_parse_weighted_formula():
    text = """
		sum(Burglary, Earthquake):
			(
				(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
			) times (
				p(Burglary)_probability times p(Earthquake)_probability
			)
		"""
    assert parse_formula(text) == _weighted_formula()


def test_comments_and_whitespace_are_ignored():
    text = """
		# Leading comment
		sum(Burglary):
			# inline comment
			=(Burglary,true)_boolean
		"""
    assert str(parse_formula(text)) == "sum(Burglary): =(Burglary,true)_boolean"


def test_parse_parenthesized_subexpression():
    text = "sum(Burglary): (=(Burglary,true)_boolean)"
    assert str(parse_formula(text)) == "sum(Burglary): =(Burglary,true)_boolean"


def test_parse_structure_alias_short_boolean():
    assert str(parse_formula("foo_b")) == "foo_boolean"


def test_parse_structure_alias_short_probability_transformation():
    assert str(parse_formula("(foo_boolean)_p")) == "(foo_boolean)_probability"


# --- Error cases ---


def test_invalid_leaf_missing_structure():
    with pytest.raises(LexError):
        str(parse_formula("=(Burglary,true)_"))


def test_reject_trailing_characters():
    with pytest.raises(ParseError):
        str(parse_formula("=(Burglary,true)_boolean junk"))


# --- Module construction ---


def test_parse_formula_with_the_additive_identity():
    """``false`` in a formula is the additive identity, not an input."""
    text = "=(X,true)_boolean and false_boolean"
    module = parse_formula_to_module(text)

    # X AND false = false for any X
    for val in [0.0, 1.0]:
        result = module(torch.tensor([[val]]))
        torch.testing.assert_close(result, torch.tensor([[0.0]], dtype=result.dtype))


def test_parse_formula_with_the_multiplicative_identity():
    """``true`` in a formula is the multiplicative identity, not an input."""
    text = "=(X,true)_boolean and true_boolean"
    module = parse_formula_to_module(text)

    # X AND true = X
    for val in [0.0, 1.0]:
        result = module(torch.tensor([[val]]))
        torch.testing.assert_close(result, torch.tensor([[val]], dtype=result.dtype))


def test_parse_formula_with_probability_constant():
    """A numeric constant in probability structure is used as a literal value."""
    text = "(=(X,true)_boolean)_probability times 0.75_probability"
    module = parse_formula_to_module(text)

    # =(X,true) is 1 when X=true, 0 when X=false
    result_true = module(torch.tensor([[1.0]]))
    assert result_true.item() == pytest.approx(0.75)

    result_false = module(torch.tensor([[0.0]]))
    assert result_false.item() == pytest.approx(0.0)


def test_parse_formula_to_module_returns_module():
    text = """
sum(Burglary, Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_real
"""
    module = parse_formula_to_module(text)
    result = module()

    torch.testing.assert_close(result, torch.tensor([[3.0]], dtype=result.dtype))


# --- Bare-atom leaves under a custom structure (Item B / C) ---


def _fuzzy_compiler(name: str = "ftest"):
    """A compiler with a single custom fuzzy AlgebraicStructure."""
    from deeplog import AlgebraicStructure
    from deeplog import Compiler

    structure = AlgebraicStructure(
        name=name,
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "not": lambda x: 1.0 - x,
        },
    )
    return Compiler(structures={name: structure})


@pytest.mark.parametrize(
    "text",
    [
        "x0_ftest or x1_ftest",
        "not x0_ftest and not x1_ftest and x2_ftest",
        "(not x0_ftest and not x1_ftest and x2_ftest) or "
        "(x0_ftest and x1_ftest and x2_ftest)",
        "(x0_ftest or x1_ftest or x2_ftest) and "
        "(not x0_ftest or not x1_ftest or x2_ftest)",
        "(x0_ftest and not x1_ftest and not x2_ftest) or "
        "(not x0_ftest and x1_ftest and not x2_ftest)",
        "(x0_ftest or x1_ftest or x2_ftest) and (not x0_ftest or not x1_ftest) and "
        "(not x1_ftest or not x2_ftest)",
    ],
)
def test_parse_bare_atom_leaves_under_custom_structure(text):
    """Bare-atom leaves (``name_struct``, no predicate args) parse correctly.

    Previously the Lark grammar left the leaf-vs-operator reading ambiguous and
    a bare leaf in operator position was mis-read as an operator IDENT
    (``ValueError: Unknown operator 'x1_ftest'``).
    """
    module = parse_formula_to_module(text, _fuzzy_compiler())
    n_leaves = len(list(module.get_input_shape()))
    out = module(torch.rand(4, n_leaves))
    assert out.shape[0] == 4


def test_custom_or_fn_honored_through_front_door():
    """Item A end-to-end: a custom fuzzy OR (a + b - a*b) parsed via the text
    front door must be honored, not replaced by the semiring sum a + b.
    """
    module = parse_formula_to_module("x0_fuzzy or x1_fuzzy", _fuzzy_compiler("fuzzy"))
    out = module(torch.tensor([[0.2, 0.7]]))
    assert float(out) == pytest.approx(0.76)  # not 0.9


def test_bare_input_leaf_compiles_to_identity():
    """Item C: a formula that is a single bare input leaf compiles (via a
    one-node circuit) to a pass-through identity module instead of raising.
    """
    module = parse_formula_to_module("x0_fuzzy", _fuzzy_compiler("fuzzy"))
    assert float(module(torch.tensor([[0.3]]))) == pytest.approx(0.3)
    torch.testing.assert_close(
        module(torch.tensor([[0.3], [0.9]])).reshape(-1),
        torch.tensor([0.3, 0.9]),
    )


def test_bare_numeric_leaf_compiles_to_constant():
    """Item C: a formula that is a single numeric symbol compiles (via the same
    one-node circuit path) to a constant module, resolved by ``constant_fn``.
    """
    from deeplog import SymTensor

    module = parse_formula_to_module("1.0_fuzzy", _fuzzy_compiler("fuzzy"))
    assert module.get_input_shape() == SymTensor([])
    torch.testing.assert_close(module(torch.zeros(3, 0)), torch.ones(3, 1))

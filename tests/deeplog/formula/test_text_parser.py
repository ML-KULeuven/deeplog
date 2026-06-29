#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch
from lark import LexError
from lark import ParseError

from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula import parse_formula
from deeplog.formula import parse_formula_to_module


def _build_weighted_formula(factory: SymbolicFormulaFactory):
    burglary = ("Burglary",)
    earthquake = ("Earthquake",)
    true = ("true",)

    b_boolean = factory.create_atom(("_", ("=", burglary, true), ("boolean",)))
    e_boolean = factory.create_atom(("_", ("=", earthquake, true), ("boolean",)))
    boolean_disjunction = factory.create_binary_node("or", b_boolean, e_boolean)
    transformed = factory.create_transformation("probability", boolean_disjunction)

    b_prob = factory.create_atom(("_", ("p", burglary), ("probability",)))
    e_prob = factory.create_atom(("_", ("p", earthquake), ("probability",)))
    joint_prob = factory.create_binary_node("times", b_prob, e_prob)

    product = factory.create_binary_node("times", transformed, joint_prob)
    return factory.create_aggregation("sum", [burglary, earthquake], [], product)


# --- Canonical round-trip: parse_formula(text) == text ---


def test_parse_leaf():
    text = "=(Burglary,true)_boolean"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


def test_parse_unary_expression():
    text = "sum(Burglary): not =(Burglary,true)_boolean"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


def test_parse_general_aggregation():
    text = "product(X): p(X)_probability"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


def test_parse_aggregation_with_params():
    text = "expect(X; q(X)_probability): p(X)_probability"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


def test_parse_transformation_only():
    text = "(=(Burglary,true)_boolean)_probability"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


def test_parse_binary_of_transformations():
    text = "(=(Burglary,true)_boolean)_probability times (=(Earthquake,true)_boolean)_probability"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


def test_parse_unary_over_transformation():
    text = "sum(Burglary): not (=(Burglary,true)_boolean)_probability"
    assert parse_formula(text, SymbolicFormulaFactory()) == text


# --- Normalization: non-canonical input → canonical output ---


def test_parse_model_count_formula():
    text = """
		sum(Burglary, Earthquake):
			=(Burglary,true)_boolean or =(Earthquake,true)_boolean
		"""
    assert (
        parse_formula(text, SymbolicFormulaFactory())
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
    factory = SymbolicFormulaFactory()
    assert parse_formula(text, factory) == _build_weighted_formula(factory)


def test_comments_and_whitespace_are_ignored():
    text = """
		# Leading comment
		sum(Burglary):
			# inline comment
			=(Burglary,true)_boolean
		"""
    assert (
        parse_formula(text, SymbolicFormulaFactory())
        == "sum(Burglary): =(Burglary,true)_boolean"
    )


def test_parse_parenthesized_subexpression():
    text = "sum(Burglary): (=(Burglary,true)_boolean)"
    assert (
        parse_formula(text, SymbolicFormulaFactory())
        == "sum(Burglary): =(Burglary,true)_boolean"
    )


def test_parse_structure_alias_short_boolean():
    assert parse_formula("foo_b", SymbolicFormulaFactory()) == "foo_boolean"


def test_parse_structure_alias_short_probability_transformation():
    assert (
        parse_formula("(foo_boolean)_p", SymbolicFormulaFactory())
        == "(foo_boolean)_probability"
    )


# --- Error cases ---


def test_invalid_leaf_missing_structure():
    with pytest.raises(LexError):
        parse_formula("=(Burglary,true)_", SymbolicFormulaFactory())


def test_reject_trailing_characters():
    with pytest.raises(ParseError):
        parse_formula("=(Burglary,true)_boolean junk", SymbolicFormulaFactory())


# --- Module construction ---


def test_parse_formula_with_constant_zero():
    """Constant 0 in a formula is recognized as the additive identity, not as an input."""
    text = "=(X,true)_boolean and 0_boolean"
    module = parse_formula_to_module(text)

    # X AND false = false for any X
    for val in [0.0, 1.0]:
        result = module(torch.tensor([[val]]))
        torch.testing.assert_close(result, torch.tensor([[0.0]], dtype=result.dtype))


def test_parse_formula_with_constant_one():
    """Constant 1 in a formula is recognized as the multiplicative identity, not as an input."""
    text = "=(X,true)_boolean and 1_boolean"
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
    =(Burglary,true)_boolean or =(Earthquake,true)_boolean
"""
    module = parse_formula_to_module(text)
    result = module()

    torch.testing.assert_close(result, torch.tensor([[3.0]], dtype=result.dtype))


# --- Bare-atom leaves under a custom structure (Item B / C) ---


def _fuzzy_factory(name: str = "ftest"):
    """A factory with a single custom fuzzy AlgebraicStructure."""
    from deeplog import AlgebraicStructure
    from deeplog import DeepLogModuleFactory

    structure = AlgebraicStructure(
        name=name,
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "not": lambda x: 1.0 - x,
        },
    )
    return DeepLogModuleFactory(structures={name: structure})


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
    module = parse_formula_to_module(text, _fuzzy_factory())
    n_leaves = len(list(module.get_input_shape()))
    out = module(torch.rand(4, n_leaves))
    assert out.shape[0] == 4


def test_custom_or_fn_honored_through_front_door():
    """Item A end-to-end: a custom fuzzy OR (a + b - a*b) parsed via the text
    front door must be honored, not replaced by the semiring sum a + b.
    """
    module = parse_formula_to_module("x0_fuzzy or x1_fuzzy", _fuzzy_factory("fuzzy"))
    out = module(torch.tensor([[0.2, 0.7]]))
    assert float(out) == pytest.approx(0.76)  # not 0.9


def test_bare_input_leaf_compiles_to_identity():
    """Item C: a formula that is a single bare input leaf compiles (via a
    one-node circuit) to a pass-through identity module instead of raising.
    """
    module = parse_formula_to_module("x0_fuzzy", _fuzzy_factory("fuzzy"))
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

    module = parse_formula_to_module("1.0_fuzzy", _fuzzy_factory("fuzzy"))
    assert module.get_input_shape() == SymTensor([])
    torch.testing.assert_close(module(torch.zeros(3, 0)), torch.ones(3, 1))

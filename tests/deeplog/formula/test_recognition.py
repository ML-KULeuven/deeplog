#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the expectation/WMC recognition pass."""

import pytest
import torch

from deeplog import DeepLogModuleFactory
from deeplog import reshape
from deeplog.formula import Aggregation
from deeplog.formula import Atom
from deeplog.formula import BinaryOp
from deeplog.formula import parse_formula_to_ast
from deeplog.formula import parse_formula_to_module
from deeplog.formula import recognize_expectation
from deeplog.formula import recognize_posterior
from deeplog.shape import SymTensor


# A hand-written weighted model count: a boolean formula cast to probability,
# times a factorized probability distribution.
WMC = (
    "sum(Burglary, Earthquake): "
    "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
    "times (prob(Burglary,true)_probability times prob(Earthquake,true)_probability)"
)

PB_SYM = ("_", ("prob", ("Burglary",), ("true",)), ("probability",))
PE_SYM = ("_", ("prob", ("Earthquake",), ("true",)), ("probability",))

CANONICAL_WMC = (
    "sum(Burglary, Earthquake): "
    "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
    "times (=(Burglary,true)_probability times =(Earthquake,true)_probability)"
)

BURGLARY_PROB_SYM = ("_", ("=", ("Burglary",), ("true",)), ("probability",))
EARTHQUAKE_PROB_SYM = ("_", ("=", ("Earthquake",), ("true",)), ("probability",))


def _fuzzy_factory(name: str = "fuzzy"):
    from deeplog import AlgebraicStructure

    structure = AlgebraicStructure(
        name=name,
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "not": lambda x: 1.0 - x,
        },
    )
    return DeepLogModuleFactory(structures={name: structure})


# --- AST-level: only same-atom retagged WMC is recognized ---


def test_recognize_alias_wmc_leaves_sum_unchanged():
    rec = recognize_expectation(parse_formula_to_ast(WMC))
    assert isinstance(rec, Aggregation)
    assert rec.operation == "sum"
    assert rec.binders == (("Burglary",), ("Earthquake",))
    assert not rec.params


def test_recognize_canonical_wmc_to_expectation():
    rec = recognize_expectation(parse_formula_to_ast(CANONICAL_WMC))
    assert isinstance(rec, Aggregation)
    assert rec.operation == "expectation"
    assert rec.binders == (("Burglary",), ("Earthquake",))
    assert not rec.params
    assert isinstance(rec.child, BinaryOp)
    assert rec.child.operator == "or"


# --- Behavioural: alias WMC stays enumeration; canonical WMC takes fast path ---


def test_wmc_default_passes_preserve_enumeration_semantics():
    on = reshape(parse_formula_to_module(WMC), input=SymTensor([PB_SYM, PE_SYM]))
    off = reshape(
        parse_formula_to_module(WMC, passes=()), input=SymTensor([PB_SYM, PE_SYM])
    )
    x = torch.tensor([[0.8, 0.3]])
    torch.testing.assert_close(on(x), off(x))


def test_canonical_wmc_takes_expectation_fast_path():
    module = reshape(
        parse_formula_to_module(CANONICAL_WMC),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )
    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )

    x = torch.tensor([[0.8, 0.3]])
    torch.testing.assert_close(module(x), torch.tensor([[0.86]]))


# --- Structure-awareness: custom/fuzzy structures are never rewritten ---


def test_fuzzy_sum_not_rewritten():
    ast = parse_formula_to_ast("sum(X): x0_fuzzy times x1_fuzzy")
    assert recognize_expectation(ast) == ast  # unchanged, still a plain sum


def test_custom_structure_honored_with_passes_on():
    """Default passes must not disturb a custom fuzzy operator end-to-end."""
    module = parse_formula_to_module("x0_fuzzy or x1_fuzzy", _fuzzy_factory("fuzzy"))
    out = module(torch.tensor([[0.2, 0.7]]))
    assert float(out) == pytest.approx(0.76)  # a + b - a*b, not the semiring a + b


# --- WMC-shaped sums are left as plain sums ---


NON_FACTORIZED = (
    "sum(Burglary, Earthquake): "
    "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
    "times (prob(Burglary,true)_probability plus prob(Earthquake,true)_probability)"
)


def test_non_factorized_probability_not_rewritten():
    rec = recognize_expectation(parse_formula_to_ast(NON_FACTORIZED))
    assert isinstance(rec, Aggregation)
    assert rec.operation == "sum"


def test_clean_sum_unchanged():
    clean = (
        "sum(Burglary, Earthquake): "
        "=(Burglary,true)_boolean or =(Earthquake,true)_boolean"
    )
    assert recognize_expectation(parse_formula_to_ast(clean)) == parse_formula_to_ast(
        clean
    )


# --- Behaviour stays identical with default passes on or off ---


NESTED_WMC = (
    "sum(Burglary): sum(Earthquake): "
    "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
    "times (prob(Burglary,true)_probability times prob(Earthquake,true)_probability)"
)


def test_nested_sum_not_rewritten():
    """An inner sum that leaves a variable free is not a complete model count.

    The inner ``sum(Earthquake)`` does not bind ``Burglary`` (free in the boolean
    formula), so folding it to an expectation would be wrong; both sums stay.
    """
    rec = recognize_expectation(parse_formula_to_ast(NESTED_WMC))
    assert isinstance(rec, Aggregation) and rec.operation == "sum"  # outer
    assert isinstance(rec.child, Aggregation) and rec.child.operation == "sum"  # inner


def test_nested_sum_recognition_is_behaviour_preserving():
    expected_input = SymTensor([PB_SYM, PE_SYM])
    on = reshape(parse_formula_to_module(NESTED_WMC), input=expected_input)
    off = reshape(parse_formula_to_module(NESTED_WMC, passes=()), input=expected_input)
    x = torch.tensor([[0.2, 0.6]])
    torch.testing.assert_close(on(x), off(x))


# Binders cover the free variables, but the probability atoms' arguments
# (``p(Burglary)``) do not match the boolean leaves (``=(Burglary,true)``).
NON_OVERLAPPING_ARGS = (
    "sum(Burglary, Earthquake): "
    "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
    "times (p(Burglary)_probability times p(Earthquake)_probability)"
)


def test_non_overlapping_args_not_rewritten():
    rec = recognize_expectation(parse_formula_to_ast(NON_OVERLAPPING_ARGS))
    assert isinstance(rec, Aggregation)
    assert rec.operation == "sum"


def test_non_overlapping_args_is_behaviour_preserving():
    p_burglary = ("_", ("p", ("Burglary",)), ("probability",))
    p_earthquake = ("_", ("p", ("Earthquake",)), ("probability",))
    expected_input = SymTensor([p_burglary, p_earthquake])
    on = reshape(parse_formula_to_module(NON_OVERLAPPING_ARGS), input=expected_input)
    off = reshape(
        parse_formula_to_module(NON_OVERLAPPING_ARGS, passes=()), input=expected_input
    )
    x = torch.tensor([[0.2, 0.6]])
    torch.testing.assert_close(on(x), off(x))


# --- The pass never touches an existing (explicit) expectation node ---


def test_explicit_expectation_unchanged_by_pass():
    node = Aggregation(
        "expectation",
        (("X",),),
        (),
        Atom(("_", ("=", ("X",), ("true",)), ("boolean",))),
    )
    assert recognize_expectation(node) == node


# --- Posterior: a divide of two WMCs becomes a divide of two expectations ---


CANONICAL_JOINT = (
    "sum(Burglary, Earthquake): "
    "(=(Burglary,true)_boolean and =(Earthquake,true)_boolean)_probability "
    "times (=(Burglary,true)_probability times =(Earthquake,true)_probability)"
)

CANONICAL_EVIDENCE = (
    "sum(Earthquake): "
    "(=(Earthquake,true)_boolean)_probability "
    "times (=(Earthquake,true)_probability)"
)


def _divide(numerator: str, evidence: str) -> BinaryOp:
    return BinaryOp(
        "divide", parse_formula_to_ast(numerator), parse_formula_to_ast(evidence)
    )


def test_recognize_posterior_rewrites_divide_to_expectations():
    rec = recognize_posterior(_divide(CANONICAL_JOINT, CANONICAL_EVIDENCE))
    assert isinstance(rec, BinaryOp) and rec.operator == "divide"
    assert isinstance(rec.lhs, Aggregation) and rec.lhs.operation == "expectation"
    assert isinstance(rec.rhs, Aggregation) and rec.rhs.operation == "expectation"
    # numerator boolean is q∧e, denominator boolean is the evidence conjunct
    assert isinstance(rec.lhs.child, BinaryOp) and rec.lhs.child.operator == "and"
    assert isinstance(rec.rhs.child, Atom)


def test_divide_of_wmcs_compiles_to_posterior():
    module = reshape(
        parse_formula_to_module(f"({CANONICAL_JOINT}) divide ({CANONICAL_EVIDENCE})"),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )
    # E[B∧E]=0.24, E[E]=0.3 ⇒ posterior 0.8
    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.3]])), torch.tensor([[0.8]])
    )


def test_recognize_posterior_leaves_non_conjunct_evidence_alone():
    """A divide whose evidence is not a conjunct of the numerator is not a
    posterior, so the pass leaves it as a plain ratio rather than rejecting it."""
    disjunctive = (
        "sum(Burglary, Earthquake): "
        "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
        "times (=(Burglary,true)_probability times =(Earthquake,true)_probability)"
    )
    node = _divide(disjunctive, CANONICAL_EVIDENCE)
    assert recognize_posterior(node) == node


def test_recognize_posterior_leaves_non_wmc_operand_alone():
    """An operand that is not an expectation / WMC leaves the divide untouched."""
    node = BinaryOp(
        "divide",
        Atom(BURGLARY_PROB_SYM),
        parse_formula_to_ast(CANONICAL_EVIDENCE),
    )
    assert recognize_posterior(node) == node


def test_plain_ratio_compiles_through_the_default_passes():
    """A `divide` that is no posterior still compiles — as a plain ratio.

    ``recognize_posterior`` runs in ``DEFAULT_PASSES``, so rejecting an
    unrecognized divide would make an ordinary ratio uncompilable through
    ``parse_formula_to_module``.
    """
    module = reshape(
        parse_formula_to_module(
            "=(Burglary,true)_probability divide =(Earthquake,true)_probability"
        ),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )
    torch.testing.assert_close(
        module(torch.tensor([[0.6, 0.3]])), torch.tensor([[2.0]])
    )

#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the expectation aggregation operator."""

import pytest
import torch

from deeplog import DeepLogModuleFactory
from deeplog.shape import SymTensor


FALSE = ("false",)
TRUE = ("true",)

BURGLARY = ("Burglary",)
EARTHQUAKE = ("Earthquake",)

BURGLARY_ATOM = ("=", BURGLARY, TRUE)
EARTHQUAKE_ATOM = ("=", EARTHQUAKE, TRUE)

BURGLARY_BOOL_SYM = ("_", BURGLARY_ATOM, ("boolean",))
EARTHQUAKE_BOOL_SYM = ("_", EARTHQUAKE_ATOM, ("boolean",))

BURGLARY_PROB_SYM = ("_", BURGLARY_ATOM, ("probability",))
EARTHQUAKE_PROB_SYM = ("_", EARTHQUAKE_ATOM, ("probability",))


def test_expectation_boolean_disjunction():
    """Expectation of Burglary OR Earthquake compiles to probability semiring."""
    factory = DeepLogModuleFactory()

    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", b_node, e_node)

    result_node = factory.create_aggregation(
        "expectation",
        [BURGLARY, EARTHQUAKE],
        [],  # No explicit probability params
        disjunction,
    )
    module = result_node.to_module()

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # Inputs are probability values for each atom
    # With P(B=true)=0.5, P(E=true)=0.5 (uniform):
    # E[B or E] = 0.75
    result = module(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.75]]))

    # With P(B=true)=0.8, P(E=true)=0.3:
    # E[B or E] = 1 - P(B=false)*P(E=false) = 1 - 0.2*0.7 = 0.86
    result2 = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(result2, torch.tensor([[0.86]]))


def test_expectation_single_variable():
    """Expectation over a single boolean variable."""
    factory = DeepLogModuleFactory()

    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    # Create a trivial circuit by double negation (equivalent to b_node)
    not_b = factory.create_unary_node("not", b_node)
    b_circuit = factory.create_unary_node("not", not_b)

    result_node = factory.create_aggregation(
        "expectation",
        [BURGLARY],
        [],
        b_circuit,
    )
    module = result_node.to_module()

    assert module.get_input_shape() == SymTensor([BURGLARY_PROB_SYM])
    assert len(list(module.get_output_shape())) == 1

    # E[B] with P(B=true)=0.5 -> 0.5
    result = module(torch.tensor([[0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.5]]))

    # E[B] with P(B=true)=0.7 -> 0.7
    result2 = module(torch.tensor([[0.7]]))
    torch.testing.assert_close(result2, torch.tensor([[0.7]]))


def test_expectation_conjunction():
    """Expectation of Burglary AND Earthquake."""
    factory = DeepLogModuleFactory()

    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    conjunction = factory.create_binary_node("and", b_node, e_node)

    result_node = factory.create_aggregation(
        "expectation",
        [BURGLARY, EARTHQUAKE],
        [],
        conjunction,
    )
    module = result_node.to_module()

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # E[B and E] with uniform (0.5, 0.5) = 0.25
    result = module(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.25]]))


def test_expectation_with_probability_param():
    """Expectation with a factorized probability formula remaps atoms."""
    factory = DeepLogModuleFactory()

    # Boolean formula: B OR E
    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", b_node, e_node)

    # Probability formula: p_B * p_E (atoms share arguments with boolean atoms)
    pb_sym = ("_", ("prob", BURGLARY, TRUE), ("probability",))
    pe_sym = ("_", ("prob", EARTHQUAKE, TRUE), ("probability",))
    pb_node = factory.create_atom(pb_sym)
    pe_node = factory.create_atom(pe_sym)
    prob_formula = factory.create_binary_node("times", pb_node, pe_node)

    result_node = factory.create_aggregation(
        "expectation",
        [BURGLARY, EARTHQUAKE],
        [prob_formula],
        disjunction,
    )
    module = result_node.to_module()

    # Input symbols should be the probability atoms, not the boolean atoms
    assert module.get_input_shape() == SymTensor([pb_sym, pe_sym])

    # E[B or E] = 1 - (1-p_B)(1-p_E)
    # With p_B=0.8, p_E=0.3: 1 - 0.2*0.7 = 0.86
    result = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(result, torch.tensor([[0.86]]))


def test_expectation_with_label_mapping():
    """Expectation maps boolean atoms to probability labels by argument overlap."""
    factory = DeepLogModuleFactory()

    # Boolean formula with "digit" atoms
    digit_atom_0 = ("digit", ("i1",), ("0",))
    digit_atom_1 = ("digit", ("i1",), ("1",))

    d0_node = factory.create_atom(("_", digit_atom_0, ("boolean",)))
    d1_node = factory.create_atom(("_", digit_atom_1, ("boolean",)))
    disjunction = factory.create_binary_node("or", d0_node, d1_node)

    # Probability formula with "classifier" atoms (same arguments as digit atoms)
    clf_sym_0 = ("_", ("classifier", ("i1",), ("0",)), ("probability",))
    clf_sym_1 = ("_", ("classifier", ("i1",), ("1",)), ("probability",))
    clf0_node = factory.create_atom(clf_sym_0)
    clf1_node = factory.create_atom(clf_sym_1)
    prob_formula = factory.create_binary_node("times", clf0_node, clf1_node)

    result_node = factory.create_aggregation(
        "expectation",
        [("i1",)],
        [prob_formula],
        disjunction,
    )
    module = result_node.to_module()

    # Input symbols should be classifier atoms
    assert module.get_input_shape() == SymTensor([clf_sym_0, clf_sym_1])

    # E[d0 or d1] = 1 - (1-p0)(1-p1) = 1 - 0.1*0.2 = 0.98
    result = module(torch.tensor([[0.9, 0.8]]))
    torch.testing.assert_close(result, torch.tensor([[0.98]]))


def test_expectation_too_many_params_raises_error():
    """Expectation with more than one param should raise an error."""
    factory = DeepLogModuleFactory()

    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", b_node, e_node)

    pb_node = factory.create_atom(("_", ("prob", BURGLARY, TRUE), ("probability",)))
    pe_node = factory.create_atom(("_", ("prob", EARTHQUAKE, TRUE), ("probability",)))

    with pytest.raises(ValueError, match="at most one"):
        factory.create_aggregation(
            "expectation",
            [BURGLARY],
            [pb_node, pe_node],
            disjunction,
        )


def test_expectation_non_factorized_raises_error():
    """Expectation with a non-factorized probability formula should raise."""
    factory = DeepLogModuleFactory()

    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", b_node, e_node)

    # Probability formula with plus (not factorized)
    pb_node = factory.create_atom(("_", ("prob", BURGLARY, TRUE), ("probability",)))
    pe_node = factory.create_atom(("_", ("prob", EARTHQUAKE, TRUE), ("probability",)))
    prob_formula = factory.create_binary_node("plus", pb_node, pe_node)

    with pytest.raises(ValueError, match="not fully factorized"):
        factory.create_aggregation(
            "expectation",
            [BURGLARY, EARTHQUAKE],
            [prob_formula],
            disjunction,
        )


def test_expectation_non_boolean_raises_error():
    """Expectation over non-boolean formula should raise an error."""
    factory = DeepLogModuleFactory()

    X = ("X",)
    prob_pred = ("p", X, ("_", ("0.3",), ("probability",)))
    prob_node = factory.create_atom(("_", prob_pred, ("probability",)))

    factory.variables[X] = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="boolean"):
        factory.create_aggregation(
            "expectation",
            [X],
            [],
            prob_node,
        )

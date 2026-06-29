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


def test_expectation_rejects_params():
    """Expectation no longer accepts probability-formula parameters.

    The boolean-to-probability leaf mapping is built from the engine's atom
    labels (see ``deeplog.formula.distribution.build_leaf_mapping``), so the
    old argument-overlap matching against an explicit probability formula is
    gone.
    """
    factory = DeepLogModuleFactory()

    b_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", b_node, e_node)

    pb_node = factory.create_atom(("_", ("prob", BURGLARY, TRUE), ("probability",)))

    with pytest.raises(ValueError, match="does not accept"):
        factory.create_aggregation(
            "expectation",
            [BURGLARY],
            [pb_node],
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

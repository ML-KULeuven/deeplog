#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import Algebra
from deeplog import DeepLogModuleFactory
from deeplog import SumsPredicate
from deeplog import SymTensor
from deeplog import get_network_predicate
from deeplog import reshape
from deeplog.formula.deeplogmodulefactory.nodes import CompositeCircuitNode
from deeplog.shape import get_all_symbols

from .testing_modules import IndexClassifier


FALSE = ("false",)
TRUE = ("true",)
BOOLEAN = [FALSE, TRUE]

BURGLARY = ("Burglary",)
EARTHQUAKE = ("Earthquake",)

BURGLARY_ATOM = ("=", BURGLARY, TRUE)
EARTHQUAKE_ATOM = ("=", EARTHQUAKE, TRUE)


def test_model_count():
    factory = DeepLogModuleFactory()

    burglary_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    earthquake_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", burglary_node, earthquake_node)
    summed = factory.create_aggregation("sum", [BURGLARY, EARTHQUAKE], [], disjunction)
    module = summed.to_module()
    result = module()

    torch.testing.assert_close(result, torch.tensor([[3.0]], dtype=result.dtype))


def test_missing_predicate_exposed_as_input():
    factory = DeepLogModuleFactory()

    atom1 = ("_", ("nn", ("0",)), ("probability",))
    atom2 = ("_", ("nn", ("1",)), ("probability",))

    node1 = factory.create_atom(atom1)
    node2 = factory.create_atom(atom2)
    module = factory.create_binary_node("times", node1, node2).to_module()

    assert module is not None

    assert set(get_all_symbols(module.get_input_shape())) == {atom1, atom2}

    input = torch.rand(5, 2)
    output = module(input)
    torch.testing.assert_close(output, (input[:, 0] * input[:, 1]).unsqueeze(1))


def test_weighted_model_count():
    burglary_prob = ("p", BURGLARY, ("_", ("0.8",), ("probability",)))
    earthquake_prob = ("p", EARTHQUAKE, ("_", ("0.3",), ("probability",)))

    factory = DeepLogModuleFactory()

    burglary_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    earthquake_node = factory.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    disjunction = factory.create_binary_node("or", burglary_node, earthquake_node)
    transformed_disjunction = factory.create_transformation("probability", disjunction)

    burglary_probability = factory.create_atom(("_", burglary_prob, ("probability",)))
    earthquake_probability = factory.create_atom(
        ("_", earthquake_prob, ("probability",))
    )
    joint_probability = factory.create_binary_node(
        "times", burglary_probability, earthquake_probability
    )

    weighted = factory.create_binary_node(
        "times", transformed_disjunction, joint_probability
    )
    summed = factory.create_aggregation("sum", [BURGLARY, EARTHQUAKE], [], weighted)
    module = summed.to_module()

    assert module is not None

    result = module()
    torch.testing.assert_close(result, torch.tensor([[1 - 0.2 * 0.7]]))


def test_mnist_addition():
    I1, I2 = ("I1",), ("I2",)
    N1, N2 = ("N1",), ("N2",)
    Sum = ("Sum",)

    sums_atom = ("sums", N1, N2, Sum)
    digit1_atom = ("digit", I1, N1)
    digit2_atom = ("digit", I2, N2)

    digit_domain = torch.arange(10)
    sum_domain = torch.arange(19)

    variable_domains = {
        N1: digit_domain,
        N2: digit_domain,
        Sum: sum_domain,
    }

    atom_builders = {
        SumsPredicate.get_predicate(): SumsPredicate,
        ("digit", 2, "probability"): get_network_predicate(
            "digit", 2, "probability", module=IndexClassifier(num_classes=10)
        ),
    }

    factory = DeepLogModuleFactory(
        variable_domains,
        atom_builders=atom_builders,
    )

    sums_node = factory.create_atom(("_", sums_atom, ("boolean",)))
    sums_probability = factory.create_transformation("probability", sums_node)

    digit1_node = factory.create_atom(("_", digit1_atom, ("probability",)))
    digit2_node = factory.create_atom(("_", digit2_atom, ("probability",)))
    digit_joint = factory.create_binary_node("times", digit1_node, digit2_node)

    root = factory.create_binary_node("times", sums_probability, digit_joint)
    summed = factory.create_aggregation("sum", [N1, N2], [], root)
    module = summed.to_module()
    module = reshape(module, input=SymTensor([I1, I2, Sum]))

    for i1_gt in range(10):
        for i2_gt in range(10):
            expected_result = torch.zeros(19, 1)
            for i1 in range(10):
                p1 = 0.9 if i1 == i1_gt else 0.1 / 9
                for i2 in range(10):
                    p2 = 0.9 if i2 == i2_gt else 0.1 / 9
                    expected_result[i1 + i2] += p1 * p2

            result = module(torch.tensor([[i1_gt, i2_gt, s] for s in range(19)]))
            torch.testing.assert_close(result, expected_result)


def test_create_unary_node_negation_boolean():
    burglary = ("Burglary",)
    burglary_atom = ("=", burglary, TRUE)

    factory = DeepLogModuleFactory()

    # Create leaf node and negate it
    burglary_node = factory.create_atom(("_", burglary_atom, ("boolean",)))
    negated_burglary = factory.create_unary_node("not", burglary_node)

    # Verify node and structure
    assert isinstance(negated_burglary, CompositeCircuitNode)
    assert negated_burglary.get_structure() == "boolean"

    # Aggregate and convert to module
    summed = factory.create_aggregation("sum", [burglary], [], negated_burglary)
    module = summed.to_module()
    assert module is not None
    result = module()

    # NOT(True) + NOT(False) = False + True = 1
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_create_unary_node_negation_probability():
    burglary = ("Burglary",)
    burglary_prob = ("p", burglary, ("_", ("0.7",), ("probability",)))

    factory = DeepLogModuleFactory()

    # Create probability leaf and negate it
    prob_node = factory.create_atom(("_", burglary_prob, ("probability",)))
    negated_prob = factory.create_unary_node("negate", prob_node)

    # Verify structure
    assert isinstance(negated_prob, CompositeCircuitNode)
    assert negated_prob.get_structure() == "probability"

    # Sum over all values: (1-0.3) + (1-0.7) = 0.7 + 0.3 = 1.0
    summed = factory.create_aggregation("sum", [burglary], [], negated_prob)
    module = summed.to_module()
    assert module is not None
    result = module()

    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_create_unary_node_with_binary_operations():
    burglary = ("Burglary",)
    earthquake = ("Earthquake",)

    burglary_atom = ("=", burglary, TRUE)
    earthquake_atom = ("=", earthquake, TRUE)

    factory = DeepLogModuleFactory()

    # Create: Burglary AND NOT(Earthquake)
    burglary_node = factory.create_atom(("_", burglary_atom, ("boolean",)))
    earthquake_node = factory.create_atom(("_", earthquake_atom, ("boolean",)))
    negated_earthquake = factory.create_unary_node("not", earthquake_node)
    conjunction = factory.create_binary_node("and", burglary_node, negated_earthquake)

    # Sum over all combinations
    summed = factory.create_aggregation("sum", [burglary, earthquake], [], conjunction)
    module = summed.to_module()
    assert module is not None
    result = module()

    # Count models where Burglary=True AND NOT(Earthquake)
    # B=0, E=0: False AND True = False
    # B=0, E=1: False AND False = False
    # B=1, E=0: True AND True = True ✓
    # B=1, E=1: True AND False = False
    # Result: 1 model
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_create_unary_node_double_negation():
    burglary_atom = ("=", BURGLARY, TRUE)

    factory = DeepLogModuleFactory()

    # Create NOT(NOT(Burglary))
    burglary_node = factory.create_atom(("_", burglary_atom, ("boolean",)))
    negated_once = factory.create_unary_node("not", burglary_node)
    negated_twice = factory.create_unary_node("not", negated_once)

    # Sum over all values - should be same as original
    summed = factory.create_aggregation("sum", [BURGLARY], [], negated_twice)
    module = summed.to_module()
    assert module is not None
    result = module()

    # NOT(NOT(True)) + NOT(NOT(False)) = True + False = 1
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_custom_algebraic_structure():
    """Test that DeepLogModuleFactory works with a custom AlgebraicStructure."""
    # Define a custom structure (same as probability but with a different name)
    custom_structure = Algebra(
        name="custom_prob",
        product="times",
        product_fn=lambda a, b: a * b,
        sum="plus",
        sum_fn=lambda a, b: a + b,
        negation="negate",
        negation_fn=lambda x: 1.0 - x,
    )

    factory = DeepLogModuleFactory(
        structures={"custom_prob": custom_structure},
    )

    atom1 = ("_", ("nn", ("0",)), ("custom_prob",))
    atom2 = ("_", ("nn", ("1",)), ("custom_prob",))

    node1 = factory.create_atom(atom1)
    node2 = factory.create_atom(atom2)
    module = factory.create_binary_node("times", node1, node2).to_module()

    assert module is not None

    input_data = torch.rand(5, 2)
    output = module(input_data)
    torch.testing.assert_close(
        output, (input_data[:, 0] * input_data[:, 1]).unsqueeze(1)
    )


def test_aggregation_defaults_to_boolean_domain():
    factory = DeepLogModuleFactory()

    burglary_node = factory.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    summed = factory.create_aggregation("sum", [BURGLARY], [], burglary_node)
    module = summed.to_module()
    assert module is not None
    result = module()

    # Should enumerate both False/True assignments even without explicit domain mapping
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))

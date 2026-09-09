#  Copyright (c) 2024-2026. KU Leuven
"""Compiling across an operator the backend has no node for."""

import torch

from deeplog.algebraic import PROBABILITY
from deeplog.algebraic import Semifield
from deeplog.circuit import Circuit
from deeplog.circuit.backends import select_backend
from deeplog.circuit.lower.generic import GenericCircuitModule
from deeplog.module import ColumnwiseModule
from deeplog.module.wrappers import WrappedModule
from deeplog.shape import get_all_symbols


def _compiled_circuits(module):
    """Every compiled-circuit module in ``module``'s tree."""
    return [m for m in module.modules() if isinstance(m, WrappedModule)]


def _combinations(module):
    """Every column-combination module in ``module``'s tree."""
    return [m for m in module.modules() if isinstance(m, ColumnwiseModule)]


def test_a_division_is_a_quotient_of_two_compiled_operands():
    """The root cut: nothing above it, so the combination is the whole module."""
    circuit = Circuit("probability")
    a, b = circuit.get_leaf_node(("a",)), circuit.get_leaf_node(("b",))
    module = circuit.to_module({circuit.get_operator("divide")(a, b): ("q",)})

    assert list(module.get_output_shape()) == [("_", ("q",), ("probability",))]
    torch.testing.assert_close(
        module(torch.tensor([[0.4, 0.5]])), torch.tensor([[0.8]])
    )
    # One compilation for both operands, not one each.
    assert len(_compiled_circuits(module)) == 1


def test_the_circuit_above_a_division_is_still_a_circuit():
    """A division feeds the circuit above it as an ordinary leaf.

    That is the whole point of cutting rather than giving up: ``times`` above the
    quotient is compiled, not applied as one more tensor operation.
    """
    circuit = Circuit("probability")
    a, b, k = (circuit.get_leaf_node((name,)) for name in "abk")
    quotient = circuit.get_operator("divide")(a, b)
    module = circuit.to_module({circuit.get_operator("times")(quotient, k): ("q",)})

    torch.testing.assert_close(
        module(torch.tensor([[0.4, 0.5, 0.25]])), torch.tensor([[0.2]])
    )

    (combination,) = _combinations(module)
    (cut_column,) = get_all_symbols(combination.get_output_shape())
    above = [
        compiled
        for compiled in _compiled_circuits(module)
        if cut_column in set(get_all_symbols(compiled.get_input_shape()))
    ]
    assert len(above) == 1


def test_a_division_under_a_division_splits_once_per_level():
    """Nesting is the same recursion: the operands are compiled by the same path."""
    circuit = Circuit("probability")
    a, b, k = (circuit.get_leaf_node((name,)) for name in "abk")
    inner = circuit.get_operator("divide")(a, b)
    module = circuit.to_module({circuit.get_operator("divide")(inner, k): ("q",)})

    torch.testing.assert_close(
        module(torch.tensor([[0.4, 0.5, 0.25]])), torch.tensor([[3.2]])
    )
    assert len(_combinations(module)) == 2


def test_operands_that_share_work_are_compiled_together():
    """Both operands are roots of one compilation, so what they share is shared."""
    circuit = Circuit("probability")
    a, b, x = (circuit.get_leaf_node((name,)) for name in "abx")
    times = circuit.get_operator("times")
    numerator = times(a, x)
    denominator = times(b, x)
    module = circuit.to_module(
        {circuit.get_operator("divide")(numerator, denominator): ("q",)}
    )

    # (0.4 * 0.5) / (0.8 * 0.5)
    torch.testing.assert_close(
        module(torch.tensor([[0.4, 0.8, 0.5]])), torch.tensor([[0.5]])
    )
    assert len(_compiled_circuits(module)) == 1


def test_a_division_takes_its_own_algebras_meaning():
    """Log-space division is subtraction, because the algebra says so."""
    circuit = Circuit("logprobability")
    a, b = circuit.get_leaf_node(("a",)), circuit.get_leaf_node(("b",))
    module = circuit.to_module({circuit.get_operator("divide")(a, b): ("q",)})

    torch.testing.assert_close(
        module(torch.tensor([[-1.0, -3.0]])), torch.tensor([[2.0]])
    )


def test_the_generic_evaluator_carries_a_division_as_a_node():
    """Only a backend that cannot walk an operator cuts at it.

    ``lower_generic`` dispatches out of ``operator_fns``, so a quotient is an
    ordinary node there and the circuit is compiled in one piece.
    """
    fuzzy = Semifield(name="fuzzy_semifield", product="times", sum="plus")
    circuit = Circuit(fuzzy)
    a, b = circuit.get_leaf_node(("a",)), circuit.get_leaf_node(("b",))
    module = circuit.to_module({circuit.get_operator("divide")(a, b): ("q",)})

    assert select_backend(circuit) == "generic"
    assert isinstance(module, GenericCircuitModule)
    torch.testing.assert_close(
        module(torch.tensor([[0.4, 0.5]])), torch.tensor([[0.8]])
    )


def test_a_cut_keeps_the_packed_input_contract():
    """A split circuit is still one module over one tensor of its own leaves."""
    circuit = Circuit("probability")
    a, b, k = (circuit.get_leaf_node((name,)) for name in "abk")
    quotient = circuit.get_operator("divide")(a, b)
    module = circuit.to_module({circuit.get_operator("times")(quotient, k): ("q",)})

    assert list(module.get_input_shape()) == [
        ("_", (name,), ("probability",)) for name in "abk"
    ]
    assert "divide" in PROBABILITY.operators


def test_a_cut_can_be_a_root_and_feed_the_circuit_above_it():
    """One value, one name: asked for by name *and* consumed further up.

    Naming it twice — once as the requested output, once as the leaf the circuit
    above reads — would make that circuit its own input.
    """
    circuit = Circuit("probability")
    a, b, k = (circuit.get_leaf_node((name,)) for name in "abk")
    quotient = circuit.get_operator("divide")(a, b)
    module = circuit.to_module(
        {quotient: ("ratio",), circuit.get_operator("times")(quotient, k): ("scaled",)}
    )

    assert list(module.get_output_shape()) == [
        ("_", ("ratio",), ("probability",)),
        ("_", ("scaled",), ("probability",)),
    ]
    torch.testing.assert_close(
        module(torch.tensor([[0.4, 0.5, 0.25]])), torch.tensor([[0.8, 0.2]])
    )

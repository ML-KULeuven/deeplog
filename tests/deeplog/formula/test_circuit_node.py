#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the circuit-node helpers, in particular ``to_module``'s root forms."""

import pytest
import torch

from deeplog import to_module
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.shape import SymTensor
from deeplog.symbol import with_structure


A = ("_", ("a",), ("boolean",))
B = ("_", ("b",), ("boolean",))


def _two_roots():
    """An or-root and an and-root sharing one boolean circuit."""
    factory = CircuitFactory()
    a, b = factory.create_atom(A), factory.create_atom(B)
    return factory.create_binary_node("or", a, b), factory.create_binary_node(
        "and", a, b
    )


def test_to_module_mapping_matches_positional():
    or_node, and_node = _two_roots()
    positional = to_module(or_node, and_node, names=(("or",), ("and",)))
    mapping = to_module({("or",): or_node, ("and",): and_node})

    assert positional.get_output_shape() == mapping.get_output_shape()
    assert positional.get_input_shape() == mapping.get_input_shape()
    x = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(positional(x), mapping(x))


def test_to_module_mapping_fixes_output_order():
    or_node, and_node = _two_roots()
    module = to_module({("and",): and_node, ("or",): or_node})
    # Root outputs are labelled with the compiled circuit's algebra.
    assert module.get_output_shape() == SymTensor(
        [with_structure(("and",), "boolean"), with_structure(("or",), "boolean")]
    )


def test_to_module_mapping_rejects_names():
    or_node, and_node = _two_roots()
    with pytest.raises(ValueError, match="names must be omitted"):
        to_module({("or",): or_node, ("and",): and_node}, names=(("x",), ("y",)))


def test_to_module_mapping_must_be_sole_argument():
    or_node, and_node = _two_roots()
    with pytest.raises(ValueError, match="only positional argument"):
        to_module(or_node, {("and",): and_node})


def test_one_node_named_twice_gives_two_columns():
    """Equivalent formulas compile to one node, and each name still gets a column.

    Knowledge compilation is canonical, so two roots really can be the same node
    — a query for ``alarm`` given ``alarm`` has a numerator and denominator that
    are the same value — and the outputs must not silently collapse into one.
    """
    or_node, _ = _two_roots()

    module = to_module(or_node, or_node, names=(("first",), ("second",)))

    assert list(module.get_output_shape()) == [
        ("_", ("first",), ("boolean",)),
        ("_", ("second",), ("boolean",)),
    ]
    out = module(torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(out, torch.tensor([[1.0, 1.0]]))

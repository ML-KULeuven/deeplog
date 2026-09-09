#  Copyright (c) 2024-2026. KU Leuven
"""Test that circuit constants (0, 1) map to neutral elements, not inputs."""

import torch

from deeplog.circuit import Circuit
from deeplog.circuit.backends import select_backend


def test_simple_constants_map_to_neutral_elements():
    """Constants ('0',) and ('1',) should map to neutral elements."""
    circuit = Circuit("probability")

    circuit.get_leaf_node(("v1",))
    circuit.get_leaf_node(("0",))
    circuit.get_leaf_node(("1",))

    # ('0',) and ('1',) should NOT be in leaf nodes
    assert ("0",) not in circuit.leaf_nodes
    assert ("1",) not in circuit.leaf_nodes

    # They should be the structure's identities
    assert circuit.zero_node is not None
    assert circuit.one_node is not None

    # v1 should be a leaf, exposed under its structure-tagged boundary name
    assert ("_", ("v1",), ("probability",)) in circuit.leaf_nodes


def test_structured_constants_map_to_neutral_elements():
    """Structured constants ('_', ('0',), ('structure',)) should map to neutral elements."""
    circuit = Circuit("probability")

    circuit.get_leaf_node(("v1",))

    # These are the actual symbol formats used in formulas
    circuit.get_leaf_node(("_", ("0",), ("probability",)))
    circuit.get_leaf_node(("_", ("1",), ("probability",)))

    # These should NOT be in leaf nodes (they should map to neutral elements)
    assert ("_", ("0",), ("probability",)) not in circuit.leaf_nodes
    assert ("_", ("1",), ("probability",)) not in circuit.leaf_nodes

    # They should be the structure's identities
    assert circuit.zero_node is not None
    assert circuit.one_node is not None


def test_get_symbol_name_decodes_leaves_and_constants():
    """Circuit-level symbol naming covers formula atoms and constants."""
    circuit = Circuit("probability")

    v1 = circuit.get_leaf_node(("v1",))
    zero = circuit.get_leaf_node(("_", ("0",), ("probability",)))
    half = circuit.get_leaf_node(("_", ("0.5",), ("probability",)))
    product = circuit.get_operator("times")(v1, half)

    assert circuit.get_symbol_name(v1) == ("_", ("v1",), ("probability",))
    assert circuit.get_symbol_name(zero) == ("_", ("0",), ("probability",))
    assert circuit.get_symbol_name(half) == ("_", ("0.5",), ("probability",))
    assert circuit.get_symbol_name(product) is None
    assert circuit.reachable_symbol_names([product]) == [
        ("_", ("v1",), ("probability",)),
        ("_", ("0.5",), ("probability",)),
    ]


def test_circuit_with_constants_has_correct_input_shape():
    """A circuit using constants should not have them in input shape."""
    circuit = Circuit("probability")

    v1 = circuit.get_leaf_node(("v1",))
    v2 = circuit.get_leaf_node(("v2",))
    const_0 = circuit.get_leaf_node(("_", ("0",), ("probability",)))

    # Build: v1 * (v2 + 0)
    or_op = circuit.get_operator("plus")
    and_op = circuit.get_operator("times")

    v2_or_0 = or_op(v2, const_0)
    result = and_op(v1, v2_or_0)

    roots = {result: ("result",)}
    module = circuit.to_module(roots)

    # Input shape should only have v1 and v2, NOT the constant. Leaves are
    # exposed under their structure-tagged boundary names.
    input_symbols = list(module.get_input_shape())
    assert len(input_symbols) == 2
    assert ("_", ("v1",), ("probability",)) in input_symbols
    assert ("_", ("v2",), ("probability",)) in input_symbols


def test_klay_carries_a_numeric_constant_as_a_prefilled_slot():
    """Klay has no constant primitive, so a constant becomes a pre-filled slot.

    The circuit keeps the Klay fast path — one numeric constant used to demote
    the whole thing to the generic evaluator — and the constant stays out of the
    module's input shape.
    """
    circuit = Circuit("probability")
    v1 = circuit.get_leaf_node(("v1",))
    half = circuit.get_leaf_node(("_", ("0.5",), ("probability",)))
    result = circuit.get_operator("times")(v1, half)

    assert select_backend(circuit) == "klay"
    module = circuit.to_module({result: ("result",)})

    assert list(module.get_input_shape()) == [("_", ("v1",), ("probability",))]
    assert torch.allclose(module(torch.tensor([[0.8]])), torch.tensor([[0.4]]))


def test_a_shared_constant_node_multiplies_once_per_use():
    """Constants dedup by value, but Klay evaluates arithmetically, not by WMC.

    Both ``0.5``s are the same node and hence one input slot; the slot's value is
    read once per parent, so ``v1 * 0.5 * 0.5`` is ``0.2`` rather than the
    ``0.4`` an idempotent conjunction would give.
    """
    circuit = Circuit("probability")
    v1 = circuit.get_leaf_node(("v1",))
    first = circuit.get_leaf_node(("_", ("0.5",), ("probability",)))
    second = circuit.get_leaf_node(("_", ("0.5",), ("probability",)))
    assert first == second

    times = circuit.get_operator("times")
    result = times(times(v1, first), second)
    module = circuit.to_module({result: ("result",)})

    assert torch.allclose(module(torch.tensor([[0.8]])), torch.tensor([[0.2]]))

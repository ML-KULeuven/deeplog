#  Copyright (c) 2024-2026. KU Leuven
"""Test that circuit constants (0, 1) map to neutral elements, not inputs."""

from deeplog.circuit import Circuit


def test_simple_constants_map_to_neutral_elements():
    """Constants ('0',) and ('1',) should map to neutral elements."""
    circuit = Circuit("probability")

    circuit.get_leaf_node(("v1",))
    circuit.get_leaf_node(("0",))
    circuit.get_leaf_node(("1",))

    # ('0',) and ('1',) should NOT be in leaf nodes
    assert ("0",) not in circuit.leaf_nodes
    assert ("1",) not in circuit.leaf_nodes

    # They should be in neutral nodes
    assert "zero" in circuit.constant_nodes
    assert "one" in circuit.constant_nodes

    # v1 should be a leaf
    assert ("v1",) in circuit.leaf_nodes


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

    # They should be in neutral nodes
    assert "zero" in circuit.constant_nodes
    assert "one" in circuit.constant_nodes


def test_circuit_with_constants_has_correct_input_shape():
    """A circuit using constants should not have them in input shape."""
    circuit = Circuit("probability", deterministic=True)

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

    # Input shape should only have v1 and v2, NOT the constant
    input_symbols = list(module.get_input_shape())
    assert len(input_symbols) == 2
    assert ("v1",) in input_symbols
    assert ("v2",) in input_symbols


if __name__ == "__main__":
    test_simple_constants_map_to_neutral_elements()
    print("test_simple_constants_map_to_neutral_elements PASSED")

    test_structured_constants_map_to_neutral_elements()
    print("test_structured_constants_map_to_neutral_elements PASSED")

    test_circuit_with_constants_has_correct_input_shape()
    print("test_circuit_with_constants_has_correct_input_shape PASSED")

#  Copyright (c) 2024-2026. KU Leuven
"""Tests for probability distribution helpers."""

import pytest

from deeplog.circuit import Circuit
from deeplog.formula import DeepLogModuleFactory
from deeplog.formula.deeplogmodulefactory.nodes import CompositeCircuitNode
from deeplog.formula.distribution import build_leaf_mapping
from deeplog.formula.distribution import build_probability_distribution
from deeplog.formula.distribution import factorize


def _make_probability_circuit(*leaf_names):
    """Create a probability circuit that is a product of the given leaf symbols."""
    circuit = Circuit("probability")
    if not leaf_names:
        return circuit, circuit.get_leaf_node(("1",))
    nodes = [circuit.get_leaf_node(name) for name in leaf_names]
    times = circuit.get_operator("times")
    root = nodes[0]
    for n in nodes[1:]:
        root = times(root, n)
    return circuit, root


class TestFactorize:
    def test_single_leaf(self):
        sym = ("_", ("a", ("x",)), ("probability",))
        circuit, root = _make_probability_circuit(sym)
        result = factorize(circuit, root)
        assert result == [sym]

    def test_product_of_leaves(self):
        sym_a = ("_", ("a", ("x",)), ("probability",))
        sym_b = ("_", ("b", ("y",)), ("probability",))
        circuit, root = _make_probability_circuit(sym_a, sym_b)
        result = factorize(circuit, root)
        assert set(result) == {sym_a, sym_b}

    def test_three_leaves(self):
        syms = [
            ("_", ("a", ("x",)), ("probability",)),
            ("_", ("b", ("y",)), ("probability",)),
            ("_", ("c", ("z",)), ("probability",)),
        ]
        circuit, root = _make_probability_circuit(*syms)
        result = factorize(circuit, root)
        assert set(result) == set(syms)

    def test_non_factorized_plus(self):
        circuit = Circuit("probability")
        a = circuit.get_leaf_node(("_", ("a", ("x",)), ("probability",)))
        b = circuit.get_leaf_node(("_", ("b", ("y",)), ("probability",)))
        plus = circuit.get_operator("plus")
        root = plus(a, b)
        with pytest.raises(ValueError, match="not fully factorized"):
            factorize(circuit, root)

    def test_non_factorized_negate(self):
        circuit = Circuit("probability")
        a = circuit.get_leaf_node(("_", ("a", ("x",)), ("probability",)))
        negate = circuit.get_operator("negate")
        root = negate(a)
        with pytest.raises(ValueError, match="not fully factorized"):
            factorize(circuit, root)


class TestBuildLeafMapping:
    def test_matching_arguments(self):
        bool_leaves = [
            ("_", ("digit", ("i1",), ("0",)), ("boolean",)),
            ("_", ("digit", ("i1",), ("1",)), ("boolean",)),
        ]
        prob_leaves = [
            ("_", ("classifier", ("i1",), ("0",)), ("probability",)),
            ("_", ("classifier", ("i1",), ("1",)), ("probability",)),
        ]
        mapping = build_leaf_mapping(bool_leaves, prob_leaves)
        assert mapping(bool_leaves[0]) == prob_leaves[0]
        assert mapping(bool_leaves[1]) == prob_leaves[1]

    def test_unmatched_boolean_maps_to_one(self):
        bool_leaves = [
            ("_", ("digit", ("i1",), ("0",)), ("boolean",)),
            ("_", ("fact", ("y",)), ("boolean",)),
        ]
        prob_leaves = [
            ("_", ("classifier", ("i1",), ("0",)), ("probability",)),
        ]
        mapping = build_leaf_mapping(bool_leaves, prob_leaves)
        assert mapping(bool_leaves[0]) == prob_leaves[0]
        assert mapping(bool_leaves[1]) == ("1",)

    def test_ambiguous_arguments_raises(self):
        bool_leaves = [("_", ("a", ("x",)), ("boolean",))]
        prob_leaves = [
            ("_", ("p1", ("x",)), ("probability",)),
            ("_", ("p2", ("x",)), ("probability",)),
        ]
        with pytest.raises(ValueError, match="Ambiguous"):
            build_leaf_mapping(bool_leaves, prob_leaves)

    def test_unknown_symbol_passes_through(self):
        mapping = build_leaf_mapping([], [])
        unknown = ("unknown",)
        assert mapping(unknown) == unknown


class TestBuildProbabilityDistribution:
    def test_empty(self):
        factory = DeepLogModuleFactory()
        assert build_probability_distribution({}, factory) is None

    def test_single_label(self):
        """A single label still produces a circuit node so downstream
        consumers can access ``.circuit`` / ``.node``."""
        factory = DeepLogModuleFactory()
        node = build_probability_distribution({("a",): ("la",)}, factory)
        assert isinstance(node, CompositeCircuitNode)

    def test_multiple_labels(self):
        """Multiple labels combine into a product circuit node."""
        factory = DeepLogModuleFactory()
        labels = {("a",): ("la",), ("b",): ("lb",)}
        node = build_probability_distribution(labels, factory)
        assert isinstance(node, CompositeCircuitNode)

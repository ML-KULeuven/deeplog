#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the Circuit class."""

import pytest
import torch

from deeplog.circuit import Circuit


class TestCircuitConstruction:
    """Test circuit construction: leaves, operators, roots."""

    def test_create_circuit_with_structure(self):
        circuit = Circuit("boolean")
        assert circuit.structure == "boolean"
        assert len(circuit.leaf_nodes) == 0

    def test_leaf_nodes_are_deduplicated(self):
        circuit = Circuit("boolean")
        node1 = circuit.get_leaf_node(("x",))
        node2 = circuit.get_leaf_node(("x",))
        assert node1 == node2
        assert len(circuit.leaf_nodes) == 1

    def test_neutral_elements_not_tracked_as_leaves(self):
        circuit = Circuit("boolean")
        circuit.get_leaf_node(("0",))  # false
        circuit.get_leaf_node(("1",))  # true
        circuit.get_leaf_node(("a",))
        assert list(circuit.leaf_nodes.keys()) == [("a",)]

    def test_operators_create_new_nodes(self):
        circuit = Circuit("boolean")
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        result = circuit.get_operator("and")(a, b)
        assert result != a and result != b


class TestCircuitToModule:
    """Test circuit.to_module() produces correct outputs."""

    @pytest.mark.parametrize(
        "structure,op,inputs,expected",
        [
            ("boolean", "and", [1.0, 1.0], 1.0),
            ("boolean", "and", [1.0, 0.0], 0.0),
            ("boolean", "or", [0.0, 0.0], 0.0),
            ("boolean", "or", [1.0, 0.0], 1.0),
            ("probability", "times", [0.5, 0.6], 0.3),
            ("probability", "plus", [0.3, 0.4], 0.7),
        ],
    )
    def test_binary_operators(self, structure, op, inputs, expected):
        circuit = Circuit(structure)
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        result = circuit.get_operator(op)(a, b)
        module = circuit.to_module({result: ("result",)})
        output = module(torch.tensor([inputs], dtype=torch.float32))
        assert output[0].item() == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize(
        "structure,op,input_val,expected",
        [
            ("boolean", "not", 1.0, 0.0),
            ("boolean", "not", 0.0, 1.0),
            ("probability", "negate", 0.7, 0.3),
            ("probability", "negate", 0.2, 0.8),
        ],
    )
    def test_unary_operators(self, structure, op, input_val, expected):
        circuit = Circuit(structure)
        x = circuit.get_leaf_node(("x",))
        result = circuit.get_operator(op)(x)
        module = circuit.to_module({result: ("result",)})
        output = module(torch.tensor([[input_val]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(expected, rel=1e-5)

    def test_neutral_elements(self):
        """true AND x = x, false OR x = x."""
        circuit = Circuit("boolean")
        a = circuit.get_leaf_node(("a",))
        true_node = circuit.get_leaf_node(("1",))
        result = circuit.get_operator("and")(a, true_node)
        module = circuit.to_module({result: ("result",)})
        for val in [0.0, 1.0]:
            output = module(torch.tensor([[val]], dtype=torch.float32))
            assert output[0].item() == pytest.approx(val)

    def test_nested_operations(self):
        """(a AND b) OR c."""
        circuit = Circuit("boolean")
        a, b, c = [circuit.get_leaf_node((x,)) for x in "abc"]
        and_op, or_op = circuit.get_operator("and"), circuit.get_operator("or")
        result = or_op(and_op(a, b), c)
        module = circuit.to_module({result: ("result",)})

        test_cases = [
            ([1.0, 1.0, 0.0], 1.0),  # (1 AND 1) OR 0 = 1
            ([0.0, 1.0, 0.0], 0.0),  # (0 AND 1) OR 0 = 0
            ([0.0, 0.0, 1.0], 1.0),  # (0 AND 0) OR 1 = 1
        ]
        for inputs, expected in test_cases:
            output = module(torch.tensor([inputs], dtype=torch.float32))
            assert output[0].item() == pytest.approx(expected)

    def test_deep_klay_chain_does_not_hit_recursion_limit(self):
        circuit = Circuit("boolean")
        a = circuit.get_leaf_node(("a",))
        result = a
        and_op = circuit.get_operator("and")

        for _ in range(1100):
            result = and_op(result, a)

        module = circuit.to_module({result: ("result",)})
        output = module(torch.tensor([[1.0]], dtype=torch.float32))

        assert output[0].item() == pytest.approx(1.0)


class TestLogProbabilityStructure:
    """Test logprobability semiring operations."""

    def test_times_is_addition_in_log_space(self):
        circuit = Circuit("logprobability")
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        result = circuit.get_operator("times")(a, b)
        module = circuit.to_module({result: ("result",)})

        probs = torch.tensor([[0.6, 0.2]], dtype=torch.float32)
        output = module(torch.log(probs))
        expected = torch.log(torch.tensor(0.6 * 0.2))
        assert output[0].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_plus_is_logsumexp(self):
        circuit = Circuit("logprobability")
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        result = circuit.get_operator("plus")(a, b)
        module = circuit.to_module({result: ("result",)})

        probs = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
        output = module(torch.log(probs))
        expected = torch.log(torch.tensor(0.4 + 0.3))
        assert output[0].item() == pytest.approx(expected.item(), rel=1e-5)


class TestGenericEvaluator:
    """Test the pure-PyTorch fallback for custom operator structures."""

    def _make_fuzzy_circuit(self):
        from deeplog.algebraic import AlgebraicStructure

        fuzzy = AlgebraicStructure(
            name="fuzzy",
            operator_fns={
                "and": lambda a, b: a * b,
                "or": lambda a, b: a + b - a * b,
                "not": lambda x: 1.0 - x,
                "implies": lambda a, b: 1.0 - a + a * b,
            },
        )
        circuit = Circuit(fuzzy)
        return circuit, fuzzy

    def test_implies_operator(self):
        """Fuzzy implication: 1 - a + a*b (LTN-style)."""
        circuit, _ = self._make_fuzzy_circuit()
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        impl = circuit.get_operator("implies")(a, b)
        module = circuit.to_module({impl: ("result",)})

        # implies(1, 0) = 0, implies(0, 0) = 1, implies(0.5, 0.5) = 0.75
        inputs = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
        out = module(inputs)
        torch.testing.assert_close(out[:, 0], torch.tensor([0.0, 1.0, 0.75]))

    def test_custom_operator_uses_generic_module(self):
        """Circuits with non-klay operators produce a GenericCircuitModule."""
        from deeplog.circuit.generic import GenericCircuitModule

        circuit, _ = self._make_fuzzy_circuit()
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        impl = circuit.get_operator("implies")(a, b)
        module = circuit.to_module({impl: ("result",)})
        assert isinstance(module, GenericCircuitModule)

    def test_generic_and_operator(self):
        """Product t-norm via generic path.

        Even though 'and' is klay-compatible, the fuzzy structure also defines
        'implies' which makes the entire structure klay-incompatible.
        """
        circuit, _ = self._make_fuzzy_circuit()
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        result = circuit.get_operator("and")(a, b)
        module = circuit.to_module({result: ("result",)})

        inputs = torch.tensor([[0.5, 0.4]])
        out = module(inputs)
        torch.testing.assert_close(out[:, 0], torch.tensor([0.2]), atol=1e-5, rtol=1e-5)

    def test_multi_root_output(self):
        """Multiple roots produce a (batch, n_roots) tensor."""
        circuit, _ = self._make_fuzzy_circuit()
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        impl = circuit.get_operator("implies")(a, b)
        conj = circuit.get_operator("and")(a, b)
        module = circuit.to_module({impl: ("impl",), conj: ("conj",)})
        inputs = torch.tensor([[0.5, 0.5]])
        out = module(inputs)
        assert out.shape == (1, 2)
        # implies(0.5, 0.5) = 0.75, and(0.5, 0.5) = 0.25
        torch.testing.assert_close(out[0, 0], torch.tensor(0.75))
        torch.testing.assert_close(out[0, 1], torch.tensor(0.25))

    def test_structure_override_raises_for_generic(self):
        """structure_override is not supported for generic evaluator."""
        circuit, _ = self._make_fuzzy_circuit()
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        impl = circuit.get_operator("implies")(a, b)
        with pytest.raises(ValueError, match="structure_override is not supported"):
            circuit.to_module({impl: ("result",)}, structure_override="probability")

    def test_klay_named_ops_on_bare_structure_use_generic(self):
        """Item A: a plain AlgebraicStructure whose operators are named like the
        klay/semiring ops (and/or/not) must still compile via the generic
        per-operator evaluator so its custom operator_fns are honored — not
        silently replaced by the semiring sum/product.
        """
        from deeplog.algebraic import AlgebraicStructure
        from deeplog.circuit.generic import GenericCircuitModule

        fuzzy = AlgebraicStructure(
            name="fuzzy_named",
            operator_fns={
                "and": lambda a, b: a * b,
                "or": lambda a, b: a + b - a * b,  # non-semiring fuzzy OR
                "not": lambda x: 1.0 - x,
            },
        )
        circuit = Circuit(fuzzy)
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        disj = circuit.get_operator("or")(a, b)
        module = circuit.to_module({disj: ("result",)})

        # Routed to the generic evaluator despite {and,or,not} ⊆ KLAY_OPS.
        assert isinstance(module, GenericCircuitModule)
        out = module(torch.tensor([[0.2, 0.7]]))
        # fuzzy OR = a + b - a*b = 0.76, NOT the semiring sum a + b = 0.9.
        torch.testing.assert_close(out[:, 0], torch.tensor([0.76]), atol=1e-6, rtol=0)

    def test_single_leaf_root_is_pass_through(self):
        """Item C: a circuit whose root is just a leaf compiles to an identity
        — the degenerate case is handled on the circuit side, no special module.
        """
        circuit, _ = self._make_fuzzy_circuit()
        leaf = circuit.get_leaf_node(("a",))
        module = circuit.to_module({leaf: ("a",)})

        x = torch.tensor([[0.3], [0.9]])
        torch.testing.assert_close(module(x), x)

    def test_constant_root_compiles_to_constant_module(self):
        """Item C: a circuit whose root is a numeric constant compiles to a
        constant module (empty input shape), which composes for a batched call
        through construct_transformation just like any other generic module.
        """
        from deeplog import Sequential
        from deeplog import construct_transformation
        from deeplog.shape import SymTensor

        circuit, _ = self._make_fuzzy_circuit()
        # constant_fn resolves the numeric symbol to a constant node.
        const = circuit.get_leaf_node(("1.0",))
        assert circuit.get_node(const).node_type == "constant"
        module = circuit.to_module({const: ("c",)})
        assert module.get_input_shape() == SymTensor([])

        # Directly callable with a (batch, 0) tensor.
        torch.testing.assert_close(module(torch.zeros(3, 0)), torch.ones(3, 1))

        # And composes for a batched input: construct_transformation maps the
        # canonical variables onto the (empty) leaf order, batch flows through.
        canonical = SymTensor([("a",), ("b",)])
        composed = Sequential(
            construct_transformation(canonical, module.get_input_shape()), module
        )
        out = composed(torch.tensor([[0.2, 0.7], [0.1, 0.9]]))
        torch.testing.assert_close(out, torch.ones(2, 1))


class TestDeterministicCircuit:
    """Test deterministic=True flag (PySDD backend)."""

    def test_deterministic_produces_same_results_for_simple_circuits(self):
        """For simple circuits, deterministic and non-deterministic should match."""
        inputs = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

        non_det_circuit = Circuit("boolean")
        a = non_det_circuit.get_leaf_node(("a",))
        b = non_det_circuit.get_leaf_node(("b",))
        result = non_det_circuit.get_operator("and")(a, b)
        non_det = non_det_circuit.to_module({result: ("result",)})(inputs)

        det_circuit = Circuit("boolean", deterministic=True)
        a = det_circuit.get_leaf_node(("a",))
        b = det_circuit.get_leaf_node(("b",))
        result = det_circuit.get_operator("and")(a, b)
        det = det_circuit.to_module({result: ("result",)})(inputs)

        torch.testing.assert_close(non_det, det)

    def test_deterministic_times_same_variable_is_idempotent(self):
        """Key difference: in SDD, a * a = a (idempotent), not a^2."""
        inputs = torch.tensor([[0.8], [0.5], [0.3]], dtype=torch.float32)

        # Deterministic: a * a = a (idempotent)
        det_circuit = Circuit("probability", deterministic=True)
        a = det_circuit.get_leaf_node(("a",))
        result = det_circuit.get_operator("times")(a, a)
        det_output = det_circuit.to_module({result: ("result",)})(inputs)
        torch.testing.assert_close(det_output, inputs, rtol=1e-5, atol=1e-5)

        # Non-deterministic: a * a = a^2
        non_det_circuit = Circuit("probability")
        a = non_det_circuit.get_leaf_node(("a",))
        result = non_det_circuit.get_operator("times")(a, a)
        non_det_output = non_det_circuit.to_module({result: ("result",)})(inputs)
        torch.testing.assert_close(non_det_output, inputs**2, rtol=1e-5, atol=1e-5)

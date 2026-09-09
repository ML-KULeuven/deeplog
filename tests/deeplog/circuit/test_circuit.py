#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the Circuit class."""

import pytest
import torch

from deeplog.circuit import Circuit
from deeplog.circuit.knowledge_compile import knowledge_compile


class TestCircuitConstruction:
    """Test circuit construction: leaves, operators, roots."""

    def test_create_circuit_with_structure(self):
        circuit = Circuit("boolean")
        assert circuit.structure.name == "boolean"
        assert len(circuit.leaf_nodes) == 0

    def test_leaf_nodes_are_deduplicated(self):
        circuit = Circuit("boolean")
        node1 = circuit.get_leaf_node(("x",))
        node2 = circuit.get_leaf_node(("x",))
        assert node1 == node2
        assert len(circuit.leaf_nodes) == 1

    def test_bare_and_wrapped_spellings_dedup_to_one_node(self):
        """A leaf's structure tag agrees with the circuit, so it is redundant.

        Bare ``inner`` and wrapped ``("_", inner, (structure,))`` are the same
        atom and resolve to one shared node, regardless of which is seen first.
        """
        circuit = Circuit("boolean")
        wrapped = circuit.get_leaf_node(("_", ("digit", ("i1",)), ("boolean",)))
        bare = circuit.get_leaf_node(("digit", ("i1",)))

        assert wrapped == bare
        assert len(circuit.leaf_nodes) == 1

    def test_leaf_is_stored_bare_and_re_tagged_on_the_boundary(self):
        """Leaves are stored bare; the tag is re-appended on the way out.

        ``get_leaf_name`` exposes the bare canonical identity, while the boundary
        views (``leaf_nodes``, ``reachable_leaves``) re-append the circuit's own
        structure tag — so a predicate leaf still matches its module by shape.
        """
        circuit = Circuit("boolean")
        node = circuit.get_leaf_node(("_", ("digit", ("i1",)), ("boolean",)))

        assert circuit.get_leaf_name(node) == ("digit", ("i1",))
        assert list(circuit.leaf_nodes) == [("_", ("digit", ("i1",)), ("boolean",))]
        assert list(circuit.reachable_leaves([node])) == [
            ("_", ("digit", ("i1",)), ("boolean",))
        ]

    def test_neutral_elements_not_tracked_as_leaves(self):
        circuit = Circuit("boolean")
        circuit.get_leaf_node(("false",))
        circuit.get_leaf_node(("true",))
        circuit.get_leaf_node(("a",))
        # The leaf is exposed under its structure-tagged boundary name.
        assert list(circuit.leaf_nodes.keys()) == [("_", ("a",), ("boolean",))]

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
        true_node = circuit.get_leaf_node(("true",))
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

    @pytest.mark.parametrize("counted", [False, True])
    def test_to_module_input_shape_scoped_to_roots(self, counted):
        """A sub-root compile only takes the leaves reachable from its roots.

        Circuits are shared, so unrelated leaves may coexist in the graph; they
        must not become input slots of a module that cannot use them. True of a
        formula as written and of its knowledge-compiled counterpart, which
        carries the unrelated leaf across too.
        """
        circuit = Circuit("boolean")
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        c = circuit.get_leaf_node(("c",))  # unrelated leaf in the same circuit
        result = circuit.get_operator("and")(a, b)
        structure = "boolean"
        if counted:
            # ``c`` is carried across too, so it stays an unrelated leaf of the
            # circuit being compiled rather than being left behind.
            compiled, compiled_map = knowledge_compile(circuit, [result, c])
            circuit, node_map = compiled.transform(
                [compiled_map[result], compiled_map[c]], "probability"
            )
            result, structure = node_map[compiled_map[result]], "probability"
        module = circuit.to_module({result: ("result",)})

        assert list(module.get_input_shape()) == [
            ("_", ("a",), (structure,)),
            ("_", ("b",), (structure,)),
        ]
        output = module(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.0)

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
        from deeplog.circuit.lower.generic import GenericCircuitModule

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
        assert circuit._get_node(const).node_type == "constant"
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


class TestModelCountVsAsWritten:
    """The two readings of one graph, and what tells them apart."""

    def test_the_two_readings_agree_on_a_conjunction_of_distinct_leaves(self):
        """A product of distinct literals is the same number either way."""
        inputs = torch.tensor([[0.5, 0.6], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

        as_written = Circuit("probability")
        a = as_written.get_leaf_node(("a",))
        b = as_written.get_leaf_node(("b",))
        product = as_written.get_operator("times")(a, b)
        written = as_written.to_module({product: ("result",)})(inputs)

        boolean = Circuit("boolean")
        x = boolean.get_leaf_node(("a",))
        y = boolean.get_leaf_node(("b",))
        conjunction = boolean.get_operator("and")(x, y)
        counted_circuit, node_map = boolean.transform([conjunction], "probability")
        counted = counted_circuit.to_module({node_map[conjunction]: ("result",)})(
            inputs
        )

        torch.testing.assert_close(written, counted)

    def test_a_leaf_used_twice_is_where_counting_parts_from_arithmetic(self):
        """``a AND a`` counted is ``a``; ``a * a`` as written is ``a²``.

        Same algebra, same graph — the difference is knowledge compilation, which
        rewrites ``a ∧ a`` to ``a`` before anything reads it as arithmetic. That
        rewrite is the whole content of "this is a model count"; nothing has to
        be recorded on the circuit for it.
        """
        inputs = torch.tensor([[0.8], [0.5], [0.3]], dtype=torch.float32)

        as_written = Circuit("probability")
        a = as_written.get_leaf_node(("a",))
        squared = as_written.to_module(
            {as_written.get_operator("times")(a, a): ("result",)}
        )(inputs)
        torch.testing.assert_close(squared, inputs**2, rtol=1e-5, atol=1e-5)

        boolean = Circuit("boolean")
        x = boolean.get_leaf_node(("a",))
        conjunction = boolean.get_operator("and")(x, x)
        compiled, compiled_map = knowledge_compile(boolean, [conjunction])
        counted_circuit, node_map = compiled.transform(
            [compiled_map[conjunction]], "probability"
        )

        counted = counted_circuit.to_module(
            {node_map[compiled_map[conjunction]]: ("result",)}
        )(inputs)
        torch.testing.assert_close(counted, inputs, rtol=1e-5, atol=1e-5)


class TestCompiledDtype:
    """A compiled circuit evaluates in its input's dtype, whichever backend runs it.

    Neither backend pins a dtype of its own, so an algebra whose behaviour is
    derived from ``torch.finfo`` — the semifield division floor — sees the dtype
    the caller actually passed.
    """

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
    def test_klay_preserves_dtype(self, dtype):
        circuit = Circuit("probability")
        a = circuit.get_leaf_node(("a",))
        b = circuit.get_leaf_node(("b",))
        result = circuit.get_operator("times")(a, b)
        module = circuit.to_module({result: ("result",)})

        out = module(torch.tensor([[0.5, 0.25]], dtype=dtype))
        assert out.dtype == dtype
        assert out.item() == pytest.approx(0.125, rel=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
    def test_klay_preserves_dtype_through_a_prefilled_constant(self, dtype):
        """A numeric constant is a pre-filled input slot, so it is a dtype too."""
        circuit = Circuit("probability")
        a = circuit.get_leaf_node(("a",))
        half = circuit.get_leaf_node(("0.5",))
        result = circuit.get_operator("times")(a, half)
        module = circuit.to_module({result: ("result",)})

        out = module(torch.tensor([[0.25]], dtype=dtype))
        assert out.dtype == dtype
        assert out.item() == pytest.approx(0.125, rel=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
    def test_generic_preserves_dtype_through_a_constant(self, dtype):
        from deeplog.algebraic import AlgebraicStructure

        fuzzy = AlgebraicStructure(
            name="fuzzy_dtype",
            operator_fns={"implies": lambda a, b: 1.0 - a + a * b},
        )
        circuit = Circuit(fuzzy)
        a = circuit.get_leaf_node(("a",))
        half = circuit.get_leaf_node(("0.5",))
        result = circuit.get_operator("implies")(a, half)
        module = circuit.to_module({result: ("result",)})

        out = module(torch.tensor([[0.5]], dtype=dtype))
        assert out.dtype == dtype
        assert out.item() == pytest.approx(0.75, rel=1e-3)


class TestReachableLeafNames:
    """The leaves-only boundary: the named leaves a set of roots reads."""

    def test_only_leaves_reachable_from_the_roots_are_reported(self):
        """An unrelated atom sharing the circuit stays out of the boundary."""
        circuit = Circuit("boolean")
        b = circuit.get_leaf_node(("=", ("B",), ("true",)))
        e = circuit.get_leaf_node(("=", ("E",), ("true",)))
        circuit.get_leaf_node(("=", ("U",), ("true",)))
        root = circuit.get_operator("or")(b, e)

        assert set(circuit.reachable_leaf_names([root])) == {
            ("_", ("=", ("B",), ("true",)), ("boolean",)),
            ("_", ("=", ("E",), ("true",)), ("boolean",)),
        }

    def test_a_root_that_is_itself_a_leaf_reports_that_leaf(self):
        circuit = Circuit("boolean")
        leaf = circuit.get_leaf_node(("x",))
        assert circuit.reachable_leaf_names([leaf]) == [("_", ("x",), ("boolean",))]

    def test_constants_are_not_part_of_the_boundary(self):
        """A constant carries no leaf name, so nothing feeds it."""
        circuit = Circuit("boolean")
        x = circuit.get_leaf_node(("x",))
        false = circuit.get_leaf_node(("false",))
        root = circuit.get_operator("or")(x, false)

        assert circuit.reachable_leaf_names([root]) == [("_", ("x",), ("boolean",))]

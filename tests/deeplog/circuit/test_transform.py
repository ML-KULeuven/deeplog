#  Copyright (c) 2024-2026. KU Leuven
"""Tests for circuit transformation between algebraic structures."""

import pytest
import torch

from deeplog.algebraic import AlgebraicStructure
from deeplog.algebraic import Semiring
from deeplog.circuit import Circuit
from deeplog.circuit import CircuitNode
from deeplog.circuit import transform_nodes
from deeplog.circuit.transform import transform_circuit


class TestAutoMapping:
    """Test automatic operator mapping between Semiring/Algebra structures."""

    def test_boolean_to_probability_operators(self):
        """AND->times, OR->plus, NOT->negate."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        and_op = src.get_operator("and")
        or_op = src.get_operator("or")
        root = or_op(and_op(a, b), a)

        target, nmap = transform_circuit(src, "probability", [root])

        assert target.structure == "probability"
        root_node = target.get_node(nmap[root])
        assert root_node.node_type == "plus"
        and_mapped = target.get_node(root_node.children[0])
        assert and_mapped.node_type == "times"

    def test_boolean_to_probability_negation(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        target, nmap = transform_circuit(src, "probability", [root])

        root_node = target.get_node(nmap[root])
        assert root_node.node_type == "negate"

    def test_probability_to_logprobability(self):
        src = Circuit("probability")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("times")(a, b)

        target, nmap = transform_circuit(src, "logprobability", [root])

        assert target.structure == "logprobability"
        root_node = target.get_node(nmap[root])
        assert root_node.node_type == "times"


class TestLeafMapping:
    """Test leaf symbol remapping."""

    def test_leaf_mapping_renames_symbols(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        def rename(s):
            return (f"renamed_{s[0]}",)

        target, nmap = transform_circuit(
            src, "probability", [root], leaf_mapping=rename
        )

        leaves = target.leaf_nodes
        assert ("renamed_a",) in leaves
        assert ("renamed_b",) in leaves
        assert ("a",) not in leaves

    def test_no_leaf_mapping_preserves_names(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        target, nmap = transform_circuit(src, "probability", [root])

        leaves = target.leaf_nodes
        assert ("a",) in leaves


class TestConstants:
    """Test constant node mapping."""

    def test_zero_constant_mapped(self):
        src = Circuit("boolean")
        zero = src.get_leaf_node(("0",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("or")(zero, a)

        target, _ = transform_circuit(src, "probability", [root])

        constants = target.constant_nodes
        assert "zero" in constants

    def test_one_constant_mapped(self):
        src = Circuit("boolean")
        one = src.get_leaf_node(("1",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("and")(one, a)

        target, _ = transform_circuit(src, "probability", [root])

        constants = target.constant_nodes
        assert "one" in constants

    def test_numeric_constant_carried_over(self):
        src = Circuit("probability")
        const = src.get_leaf_node(("0.5",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("times")(const, a)

        target, nmap = transform_circuit(src, "logprobability", [root])

        assert target.constant_values[nmap[const]] == 0.5


class TestDeterministic:
    """Test deterministic flag handling."""

    def test_explicit_deterministic_true(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))

        target, _ = transform_circuit(src, "probability", [a], deterministic=True)

        assert target.deterministic is True

    def test_explicit_deterministic_false(self):
        src = Circuit("boolean", deterministic=True)
        a = src.get_leaf_node(("a",))

        target, _ = transform_circuit(src, "probability", [a], deterministic=False)

        assert target.deterministic is False

    def test_default_deterministic_from_target_non_idempotent(self):
        """Probability is not idempotent, so deterministic defaults to True."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))

        target, _ = transform_circuit(src, "probability", [a])

        assert target.deterministic is True

    def test_default_deterministic_from_target_idempotent(self):
        """Boolean is idempotent, so deterministic defaults to False."""
        src = Circuit("probability")
        a = src.get_leaf_node(("a",))

        target, _ = transform_circuit(src, "boolean", [a])

        assert target.deterministic is False


class TestExplicitOperatorMapping:
    """Test explicit operator_mapping parameter."""

    def test_custom_mapping(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        # Map "and" to "plus" instead of the default "times"
        target, nmap = transform_circuit(
            src, "probability", [root], operator_mapping={"and": "plus"}
        )

        root_node = target.get_node(nmap[root])
        assert root_node.node_type == "plus"

    def test_explicit_overrides_auto(self):
        """Explicit mapping takes precedence over auto role-based mapping."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("or")(a, b)

        target, nmap = transform_circuit(
            src, "probability", [root], operator_mapping={"or": "times"}
        )

        root_node = target.get_node(nmap[root])
        assert root_node.node_type == "times"


class TestErrors:
    """Test error handling."""

    def test_non_semiring_without_mapping_raises(self):
        custom = AlgebraicStructure(
            name="custom",
            operator_fns={"foo": lambda a, b: a + b},
        )
        src = Circuit(custom)
        a = src.get_leaf_node(("a",))
        root = src.get_operator("foo")(a, a)

        with pytest.raises(ValueError, match="operator_mapping"):
            transform_circuit(src, "probability", [root])

    def test_invalid_target_operator_raises(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        with pytest.raises(ValueError, match="nonexistent"):
            transform_circuit(
                src,
                "probability",
                [root],
                operator_mapping={"not": "nonexistent"},
            )

    def test_unmapped_operator_raises(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        and_node = src.get_operator("and")(a, b)
        root = src.get_operator("or")(and_node, a)

        # Only map "or", not "and"
        with pytest.raises(ValueError, match="and"):
            transform_circuit(
                src,
                "probability",
                [root],
                operator_mapping={"or": "plus"},
            )

    def test_algebra_to_semiring_auto_raises(self):
        """Source has negation but target is only Semiring."""
        target_struct = Semiring(
            name="test_semiring",
            product="mul",
            sum="add",
        )
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        with pytest.raises(ValueError, match="negation"):
            transform_circuit(src, target_struct, [root])


class TestNodeMap:
    """Test that the returned node_map is correct."""

    def test_node_map_covers_all_reachable_nodes(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        and_node = src.get_operator("and")(a, b)
        root = src.get_operator("or")(and_node, b)

        _, nmap = transform_circuit(src, "probability", [root])

        assert a in nmap
        assert b in nmap
        assert and_node in nmap
        assert root in nmap

    def test_shared_nodes_deduplicated(self):
        """A node used in multiple places should map to a single target node."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("and")(a, a)

        target, nmap = transform_circuit(src, "probability", [root])

        assert len(target.leaf_nodes) == 1


class TestCircuitNodeTransform:
    """Test the CircuitNode.transform_circuit() convenience method."""

    def test_circuit_node_transform(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        cn = CircuitNode(src, root)
        result = cn.transform_circuit("probability")

        assert isinstance(result, CircuitNode)
        assert result.circuit.structure == "probability"
        assert result.get_structure() == "probability"

    def test_circuit_node_transform_with_deterministic(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        cn = CircuitNode(src, root)
        result = cn.transform_circuit("probability", deterministic=True)

        assert result.circuit.deterministic is True


class TestTransformNodes:
    """Test the transform_nodes() helper for multiple CircuitNodes."""

    def test_transform_multiple_nodes(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root1 = src.get_operator("and")(a, b)
        root2 = src.get_operator("or")(a, b)

        cn1 = CircuitNode(src, root1)
        cn2 = CircuitNode(src, root2)
        r1, r2 = transform_nodes(cn1, cn2, target_structure="probability")

        assert r1.circuit is r2.circuit
        assert r1.circuit.structure == "probability"
        node1 = r1.circuit.get_node(r1.node)
        node2 = r2.circuit.get_node(r2.node)
        assert node1.node_type == "times"
        assert node2.node_type == "plus"

    def test_transform_single_node(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        (result,) = transform_nodes(
            CircuitNode(src, root), target_structure="probability"
        )

        assert result.circuit.structure == "probability"

    def test_transform_nodes_shared_leaves(self):
        """Nodes sharing leaves should share them in the target too."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root1 = src.get_operator("not")(a)
        root2 = src.get_operator("not")(root1)

        cn1 = CircuitNode(src, root1)
        cn2 = CircuitNode(src, root2)
        r1, r2 = transform_nodes(cn1, cn2, target_structure="probability")

        assert len(r1.circuit.leaf_nodes) == 1

    def test_transform_nodes_different_circuits_raises(self):
        src1 = Circuit("boolean")
        src2 = Circuit("boolean")
        a = src1.get_leaf_node(("a",))
        b = src2.get_leaf_node(("b",))

        with pytest.raises(ValueError, match="same circuit"):
            transform_nodes(
                CircuitNode(src1, a),
                CircuitNode(src2, b),
                target_structure="probability",
            )

    def test_transform_nodes_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            transform_nodes(target_structure="probability")

    def test_transform_nodes_with_deterministic(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        cn = CircuitNode(src, a)

        (result,) = transform_nodes(
            cn, target_structure="probability", deterministic=True
        )

        assert result.circuit.deterministic is True


class TestFunctional:
    """Functional tests: transform_circuit + compile + run."""

    def test_boolean_to_probability_and(self):
        """Boolean AND(a,b) -> probability times(a,b): 0.5 * 0.6 = 0.3."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        target, nmap = transform_circuit(
            src, "probability", [root], deterministic=False
        )

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.5, 0.6]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.3, rel=1e-5)

    def test_boolean_to_probability_or(self):
        """Boolean OR(a,b) -> probability plus(a,b): 0.3 + 0.4 = 0.7."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("or")(a, b)

        target, nmap = transform_circuit(
            src, "probability", [root], deterministic=False
        )

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.3, 0.4]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.7, rel=1e-5)

    def test_boolean_to_probability_deterministic(self):
        """Default deterministic=True for probability: OR uses inclusion-exclusion."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("or")(a, b)

        target, nmap = transform_circuit(src, "probability", [root])

        assert target.deterministic is True
        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.3, 0.4]], dtype=torch.float32))
        # P(a OR b) = P(a) + P(b) - P(a)*P(b) = 0.3 + 0.4 - 0.12 = 0.58
        assert output[0].item() == pytest.approx(0.58, rel=1e-5)

    def test_boolean_to_probability_not(self):
        """Boolean NOT(a) -> probability negate(a): 1.0 - 0.7 = 0.3."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        target, nmap = transform_circuit(
            src, "probability", [root], deterministic=False
        )

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.7]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.3, rel=1e-5)

    def test_boolean_to_probability_with_constants(self):
        """Boolean OR(0, a) -> probability plus(0, a) = a."""
        src = Circuit("boolean")
        zero = src.get_leaf_node(("0",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("or")(zero, a)

        target, nmap = transform_circuit(
            src, "probability", [root], deterministic=False
        )

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.42]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.42, rel=1e-5)

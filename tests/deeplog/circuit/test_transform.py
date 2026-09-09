#  Copyright (c) 2024-2026. KU Leuven
"""Tests for circuit transformation between algebraic structures."""

import pytest
import torch

from deeplog import CircuitNode
from deeplog import transform_nodes
from deeplog.algebraic import AlgebraicStructure
from deeplog.algebraic import Semiring
from deeplog.circuit import Circuit
from deeplog.circuit.knowledge_compile import knowledge_compile
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

        assert target.structure.name == "probability"
        root_node = target._get_node(nmap[root])
        assert root_node.node_type == "plus"
        and_mapped = target._get_node(root_node.children[0])
        assert and_mapped.node_type == "times"

    def test_boolean_to_probability_negation(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        target, nmap = transform_circuit(src, "probability", [root])

        root_node = target._get_node(nmap[root])
        assert root_node.node_type == "negate"

    def test_probability_to_logprobability(self):
        src = Circuit("probability")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("times")(a, b)

        target, nmap = transform_circuit(src, "logprobability", [root])

        assert target.structure.name == "logprobability"
        root_node = target._get_node(nmap[root])
        assert root_node.node_type == "times"

    def test_division_crosses_between_semifields(self):
        """A ``divide`` node crosses like any other role both structures declare."""
        src = Circuit("probability")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("times")(src.get_operator("divide")(a, b), a)

        target, nmap = transform_circuit(src, "logprobability", [root])

        root_node = target._get_node(nmap[root])
        assert root_node.node_type == "times"
        assert {target._get_node(child).node_type for child in root_node.children} == {
            "divide",
            "leaf",
        }

    def test_a_role_the_target_lacks_is_no_obstacle_when_unused(self):
        """Mapping is about the nodes there are, not about the pair of structures.

        ``boolean`` declares a negation and this target does not, but the
        transformed subgraph has no ``not`` in it, so nothing is missing.
        """
        target_struct = Semiring(name="test_semiring", product="mul", sum="add")
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        src.get_operator("not")(a)  # present in the circuit, but not under the root
        root = src.get_operator("and")(a, b)

        target, nmap = transform_circuit(src, target_struct, [root])

        assert target._get_node(nmap[root]).node_type == "mul"


class TestLeafMapping:
    """Test leaf symbol remapping."""

    def test_leaf_mapping_renames_symbols(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        def rename(s):
            # ``leaf_mapping`` sees the bare leaf identity.
            return (f"renamed_{s[0]}",)

        target, nmap = transform_circuit(
            src, "probability", [root], leaf_mapping=rename
        )

        # Leaves are exposed under their (target) structure-tagged boundary names.
        leaves = target.leaf_nodes
        assert ("_", ("renamed_a",), ("probability",)) in leaves
        assert ("_", ("renamed_b",), ("probability",)) in leaves
        assert ("_", ("a",), ("probability",)) not in leaves

    def test_no_leaf_mapping_preserves_names(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        target, nmap = transform_circuit(src, "probability", [root])

        # The bare identity is preserved, re-tagged to the target structure.
        leaves = target.leaf_nodes
        assert ("_", ("a",), ("probability",)) in leaves


class TestConstants:
    """Test constant node mapping."""

    def test_zero_constant_mapped(self):
        src = Circuit("boolean")
        zero = src.get_leaf_node(("false",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("or")(zero, a)

        target, _ = transform_circuit(src, "probability", [root])

        assert target.zero_node is not None

    def test_one_constant_mapped(self):
        src = Circuit("boolean")
        one = src.get_leaf_node(("true",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("and")(one, a)

        target, _ = transform_circuit(src, "probability", [root])

        assert target.one_node is not None

    def test_numeric_constant_carried_over(self):
        src = Circuit("probability")
        const = src.get_leaf_node(("0.5",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("times")(const, a)

        target, nmap = transform_circuit(src, "logprobability", [root])

        assert target.constant_values[nmap[const]] == 0.5


class TestStructureMap:
    """The transform maps operators onto operators, and claims nothing more."""

    def test_the_map_is_read_as_written(self):
        """``or`` becomes ``plus``, and ``plus`` is addition.

        Not a weighted model count: ``0.3 + 0.4`` is not ``P(a ∨ b)`` unless the
        two are disjoint. Making that true is knowledge compilation's job, and
        the transform is honest only when applied to its output.
        """
        source = Circuit("boolean")
        a, b = source.get_leaf_node(("a",)), source.get_leaf_node(("b",))
        root = source.get_operator("or")(a, b)

        target, node_map = transform_circuit(source, "probability", [root])

        module = target.to_module({node_map[root]: ("result",)})
        output = module(torch.tensor([[0.3, 0.4]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.7, rel=1e-5)

    def test_knowledge_compiling_first_makes_the_same_map_a_model_count(self):
        """The two rewrites together are the count: 0.3 + 0.4 - 0.12 = 0.58."""
        source = Circuit("boolean")
        a, b = source.get_leaf_node(("a",)), source.get_leaf_node(("b",))
        root = source.get_operator("or")(a, b)

        compiled, compiled_map = knowledge_compile(source, [root])
        target, node_map = transform_circuit(
            compiled, "probability", [compiled_map[root]]
        )

        module = target.to_module({node_map[compiled_map[root]]: ("result",)})
        output = module(torch.tensor([[0.3, 0.4]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.58, rel=1e-5)


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

        root_node = target._get_node(nmap[root])
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

        root_node = target._get_node(nmap[root])
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
        """A node whose role the target does not declare raises, naming it."""
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
    """Transforming a single CircuitNode via the free `transform_nodes`."""

    def test_circuit_node_transform(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        cn = CircuitNode(src, root)
        (result,) = transform_nodes(cn, target_structure="probability")

        assert isinstance(result, CircuitNode)
        assert result.circuit.structure.name == "probability"


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
        assert r1.circuit.structure.name == "probability"
        node1 = r1.circuit._get_node(r1.node)
        node2 = r2.circuit._get_node(r2.node)
        assert node1.node_type == "times"
        assert node2.node_type == "plus"

    def test_transform_single_node(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        (result,) = transform_nodes(
            CircuitNode(src, root), target_structure="probability"
        )

        assert result.circuit.structure.name == "probability"

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


class TestIncrementalTarget:
    """Test the ``into=`` entry point: accumulate several roots into one target.

    This is what lets the module factory transform each expectation eagerly yet
    have them co-reside in one circuit — equivalent to a single batched
    ``transform_nodes`` call, but spread across calls.
    """

    def test_two_calls_into_one_target_match_batched(self):
        """Two roots transformed in two calls into one target match transform_nodes."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root1 = src.get_operator("and")(a, b)
        root2 = src.get_operator("or")(a, b)

        # Incremental: transform root1, then continue root2 into the same target.
        target, nmap = transform_circuit(src, "probability", [root1])
        target2, nmap2 = transform_circuit(
            src, "probability", [root2], into=(target, nmap)
        )

        # Same objects threaded through; both roots present; leaves shared.
        assert target2 is target
        assert nmap2 is nmap
        assert root1 in nmap and root2 in nmap
        assert len(target.leaf_nodes) == 2  # a, b shared across both roots
        assert target._get_node(nmap[root1]).node_type == "times"
        assert target._get_node(nmap[root2]).node_type == "plus"

        # Batched: one transform_nodes over both roots — same operators.
        b1, b2 = transform_nodes(
            CircuitNode(src, root1),
            CircuitNode(src, root2),
            target_structure="probability",
        )
        assert b1.circuit is b2.circuit
        assert b1.circuit._get_node(b1.node).node_type == "times"
        assert b2.circuit._get_node(b2.node).node_type == "plus"

    def test_shared_subcircuit_transformed_once(self):
        """A sub-circuit shared by two roots is reused via the persistent node_map."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        shared = src.get_operator("and")(a, b)
        root1 = src.get_operator("or")(shared, a)
        root2 = src.get_operator("or")(shared, b)

        target, nmap = transform_circuit(src, "probability", [root1])
        shared_target = nmap[shared]
        # Second call must reuse the already-mapped shared node, not rebuild it.
        transform_circuit(src, "probability", [root2], into=(target, nmap))

        assert nmap[shared] == shared_target  # stable mapping
        times_nodes = [
            nid
            for nid in target.iter_topological([nmap[root1], nmap[root2]])
            if target._get_node(nid).node_type == "times"
        ]
        assert len(times_nodes) == 1  # the shared AND mapped to a single times

    def test_second_root_numerically_correct(self):
        """root2, whose sub-circuit was first mapped by root1's call, still evaluates."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        shared = src.get_operator("and")(a, b)  # times(a, b)
        root1 = src.get_operator("or")(shared, a)
        root2 = src.get_operator("or")(shared, b)  # plus(times(a, b), b)

        target, nmap = transform_circuit(src, "probability", [root1])
        transform_circuit(src, "probability", [root2], into=(target, nmap))

        # plus(times(a, b), b) with a=0.5, b=0.4 = 0.2 + 0.4 = 0.6
        module = target.to_module({nmap[root2]: ("result",)})
        output = module(torch.tensor([[0.5, 0.4]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.6, rel=1e-5)

    def test_into_structure_mismatch_raises(self):
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        prob_target, nmap = transform_circuit(src, "probability", [a])

        with pytest.raises(ValueError, match="continue a transform"):
            transform_circuit(src, "logprobability", [a], into=(prob_target, nmap))


class TestFunctional:
    """Functional tests: transform_circuit + compile + run."""

    def test_boolean_to_probability_and(self):
        """Boolean AND(a,b) -> probability times(a,b): 0.5 * 0.6 = 0.3."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("and")(a, b)

        target, nmap = transform_circuit(src, "probability", [root])

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.5, 0.6]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.3, rel=1e-5)

    def test_boolean_to_probability_or(self):
        """Boolean OR(a,b) -> probability plus(a,b): 0.3 + 0.4 = 0.7."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        b = src.get_leaf_node(("b",))
        root = src.get_operator("or")(a, b)

        target, nmap = transform_circuit(src, "probability", [root])

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.3, 0.4]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.7, rel=1e-5)

    def test_boolean_to_probability_not(self):
        """Boolean NOT(a) -> probability negate(a): 1.0 - 0.7 = 0.3."""
        src = Circuit("boolean")
        a = src.get_leaf_node(("a",))
        root = src.get_operator("not")(a)

        target, nmap = transform_circuit(src, "probability", [root])

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.7]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.3, rel=1e-5)

    def test_boolean_to_probability_with_constants(self):
        """Boolean OR(false, a) -> probability plus(zero, a) = a."""
        src = Circuit("boolean")
        zero = src.get_leaf_node(("false",))
        a = src.get_leaf_node(("a",))
        root = src.get_operator("or")(zero, a)

        target, nmap = transform_circuit(src, "probability", [root])

        module = target.to_module({nmap[root]: ("result",)})
        output = module(torch.tensor([[0.42]], dtype=torch.float32))
        assert output[0].item() == pytest.approx(0.42, rel=1e-5)

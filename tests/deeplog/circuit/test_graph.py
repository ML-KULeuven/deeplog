#  Copyright (c) 2024-2026. KU Leuven

import pytest

from deeplog.circuit.graph import Graph


class TestGraphInitialization:
    def test_create_with_node_types(self):
        graph = Graph({"and", "or", "not", "leaf"})
        assert graph.node_types == frozenset({"and", "or", "not", "leaf"})
        assert len(graph) == 0

    def test_create_with_empty_node_types(self):
        graph = Graph(set())
        assert len(graph.node_types) == 0


class TestGraphLeafNodes:
    def test_add_leaf(self):
        graph = Graph({"leaf"})
        leaf_id = graph.add_leaf()
        assert leaf_id == 0
        assert len(graph) == 1

    def test_add_multiple_leaves(self):
        graph = Graph({"leaf"})
        leaf1 = graph.add_leaf()
        leaf2 = graph.add_leaf()
        leaf3 = graph.add_leaf()
        assert leaf1 == 0
        assert leaf2 == 1
        assert leaf3 == 2
        assert len(graph) == 3

    def test_get_nodes_by_type(self):
        graph = Graph({"leaf", "and"})
        leaf1 = graph.add_leaf()
        leaf2 = graph.add_leaf()
        internal = graph.add_node("and", (leaf1, leaf2))
        leaves = graph.get_nodes_by_type("leaf")
        assert leaf1 in leaves
        assert leaf2 in leaves
        assert internal not in leaves
        assert len(leaves) == 2


class TestGraphInternalNodes:
    def test_add_node_binary(self):
        graph = Graph({"and", "or", "leaf"})
        leaf1 = graph.add_leaf()
        leaf2 = graph.add_leaf()
        node_id = graph.add_node("and", (leaf1, leaf2))
        assert node_id == 2
        assert len(graph) == 3

    def test_add_node_unary(self):
        graph = Graph({"not", "leaf"})
        leaf = graph.add_leaf()
        node_id = graph.add_node("not", (leaf,))
        assert node_id == 1

    def test_add_node_nary(self):
        graph = Graph({"nary_op", "leaf"})
        leaves = [graph.add_leaf() for _ in range(5)]
        node_id = graph.add_node("nary_op", tuple(leaves))
        assert node_id == 5
        node = graph.get_node(node_id)
        assert len(node.children) == 5

    def test_add_node_invalid_type(self):
        graph = Graph({"and", "leaf"})
        leaf = graph.add_leaf()
        with pytest.raises(ValueError, match="Unknown node_type"):
            graph.add_node("invalid", (leaf,))

    def test_add_node_invalid_child(self):
        graph = Graph({"and", "leaf"})
        leaf = graph.add_leaf()
        with pytest.raises(ValueError, match="does not exist"):
            graph.add_node("and", (leaf, 999))


class TestGraphIteration:
    def test_iter_nodes(self):
        graph = Graph({"and", "leaf"})
        leaf1 = graph.add_leaf()
        leaf2 = graph.add_leaf()
        node = graph.add_node("and", (leaf1, leaf2))
        node_ids = list(graph)
        assert leaf1 in node_ids
        assert leaf2 in node_ids
        assert node in node_ids

    def test_contains(self):
        graph = Graph({"leaf"})
        leaf = graph.add_leaf()
        assert leaf in graph
        assert 999 not in graph

    def test_iter_topological(self):
        graph = Graph({"and", "or", "leaf"})
        a = graph.add_leaf()
        b = graph.add_leaf()
        c = graph.add_leaf()
        ab = graph.add_node("and", (a, b))
        abc = graph.add_node("or", (ab, c))

        topo_order = list(graph.iter_topological([abc]))
        assert topo_order.index(a) < topo_order.index(ab)
        assert topo_order.index(b) < topo_order.index(ab)
        assert topo_order.index(ab) < topo_order.index(abc)
        assert topo_order.index(c) < topo_order.index(abc)

    def test_iter_topological_multiple_roots(self):
        graph = Graph({"and", "leaf"})
        a = graph.add_leaf()
        b = graph.add_leaf()
        c = graph.add_leaf()
        ab = graph.add_node("and", (a, b))
        bc = graph.add_node("and", (b, c))

        topo_order = list(graph.iter_topological([ab, bc]))
        assert topo_order.index(a) < topo_order.index(ab)
        assert topo_order.index(b) < topo_order.index(ab)
        assert topo_order.index(b) < topo_order.index(bc)
        assert topo_order.index(c) < topo_order.index(bc)


class TestGraphNode:
    def test_get_leaf_node(self):
        graph = Graph({"leaf"})
        leaf = graph.add_leaf()
        node = graph.get_node(leaf)
        assert node.id == leaf
        assert node.node_type == "leaf"
        assert node.children == ()

    def test_get_internal_node(self):
        graph = Graph({"and", "leaf"})
        a = graph.add_leaf()
        b = graph.add_leaf()
        internal = graph.add_node("and", (a, b))
        node = graph.get_node(internal)
        assert node.id == internal
        assert node.node_type == "and"
        assert node.children == (a, b)

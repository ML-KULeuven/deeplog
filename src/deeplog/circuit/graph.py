#  Copyright (c) 2024-2026. KU Leuven
"""Simple graph class for efficient circuit construction with integer node IDs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class Node:
    """A node in the graph with an integer ID and node type."""

    id: int
    node_type: str
    children: tuple[int, ...] = ()


@dataclass
class Graph:
    """Efficient graph structure for circuit building with integer node IDs.

    Supports n-ary nodes with configurable node types (AND, OR, NOT, LEAF, etc.).
    Designed for efficient iteration and construction before conversion to
    klay or PySDD backends.
    """

    node_types: frozenset[str]
    _nodes: dict[int, Node] = field(default_factory=dict[int, Node])
    _next_id: int = field(default=0)

    def __init__(self, node_types: set[str] | frozenset[str]) -> None:
        """Initialize a graph with the allowed node types.

        Args:
            node_types: Set of node type strings (e.g., {"and", "or", "not", "leaf"}).
        """
        self.node_types = frozenset(node_types)
        self._nodes: dict[int, Node] = {}
        self._next_id = 0

    def add_leaf(self) -> int:
        """Add a leaf node and return its ID."""
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = Node(id=node_id, node_type="leaf")
        return node_id

    def add_node(self, node_type: str, children: tuple[int, ...]) -> int:
        """Add an internal node with the given node type and children.

        Args:
            node_type: The node type (must be in self.node_types).
            children: Tuple of child node IDs.

        Returns:
            The new node's ID.

        Raises:
            ValueError: If node_type is not allowed or children are invalid.
        """
        if node_type not in self.node_types:
            raise ValueError(
                f"Unknown node_type '{node_type}'. Allowed: {self.node_types}"
            )
        for child in children:
            if child not in self._nodes:
                raise ValueError(f"Child node {child} does not exist.")

        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = Node(
            id=node_id,
            node_type=node_type,
            children=children,
        )
        return node_id

    def get_node(self, node_id: int) -> Node:
        """Return the node with the given ID."""
        return self._nodes[node_id]

    def get_nodes_by_type(self, node_type: str) -> list[int]:
        """Return all node IDs with the given node type in insertion order."""
        return [
            node_id
            for node_id, node in self._nodes.items()
            if node.node_type == node_type
        ]

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[int]:
        """Iterate over node IDs in insertion order."""
        return iter(self._nodes)

    def __contains__(self, node_id: int) -> bool:
        """Check if a node ID exists in the graph."""
        return node_id in self._nodes

    def iter_topological(self, roots: list[int]) -> Iterator[int]:
        """Iterate over nodes in topological order (leaves first).

        Args:
            roots: List of root node IDs to start traversal from.

        Yields node IDs such that all children are yielded before their parents.
        """
        visited: set[int] = set()
        stack: list[tuple[int, bool]] = [(root, False) for root in reversed(roots)]

        while stack:
            node_id, expanded = stack.pop()
            if expanded:
                yield node_id
                continue
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.append((node_id, True))
            node = self._nodes[node_id]
            for child in reversed(node.children):
                if child not in visited:
                    stack.append((child, False))

    def iter_reverse_topological(self, roots: list[int]) -> Iterator[int]:
        """Iterate over nodes in reverse topological order (roots first).

        Args:
            roots: List of root node IDs to start traversal from.

        Yields node IDs such that all parents are yielded before their children.
        """
        return reversed(list(self.iter_topological(roots)))

    def flatten_chains(
        self,
        roots: list[int],
        chain_groups: list[tuple[frozenset[str], frozenset[int]]],
    ) -> tuple[set[int], dict[int, list[int]]]:
        """Collapse chains of same-type nodes for each ``(types, absorb)`` group.

        For every node whose ``node_type`` is in ``types``: walk down through
        children whose type is also in ``types`` and whose parent count is
        exactly one, pulling their non-chain children up. Children whose id
        is in ``absorb`` are skipped entirely (intended for identity
        constants: e.g. drop the zero-id child from an OR chain).

        Args:
            roots: Roots driving the traversal and parent-counting scope.
            chain_groups: Each pair declares one set of mutually-flattening
                node types and the ids to drop while flattening them. Pass
                e.g. ``[({"and", "times"}, {one_id}), ({"or", "plus"}, {zero_id})]``.

        Returns:
            ``(absorbed, flat_children)`` where ``absorbed`` is the set of
            node ids merged into a parent (callers should skip these in any
            downstream walk), and ``flat_children`` maps each chain-head
            node id to its flattened list of non-chain children.
        """
        # Roots-scoped parent counts. Roots themselves get +1 (they're "used"
        # by the caller) so they're never absorbed. A node with parent_count
        # > 1 is shared and must remain a real intermediate rather than being
        # inlined into a parent chain.
        parents: dict[int, int] = {}
        for node_id in self.iter_topological(roots):
            for child in self._nodes[node_id].children:
                parents[child] = parents.get(child, 0) + 1
        for root_id in roots:
            parents[root_id] = parents.get(root_id, 0) + 1

        type_to_group: dict[str, tuple[frozenset[str], frozenset[int]]] = {}
        for types, absorb in chain_groups:
            for t in types:
                type_to_group[t] = (types, absorb)

        absorbed: set[int] = set()
        for node_id in self.iter_topological(roots):
            node = self._nodes[node_id]
            group = type_to_group.get(node.node_type)
            if group is None:
                continue
            chain_types, absorb_ids = group
            for child in node.children:
                if child in absorb_ids:
                    absorbed.add(child)
                elif (
                    self._nodes[child].node_type in chain_types and parents[child] == 1
                ):
                    absorbed.add(child)

        flat_children: dict[int, list[int]] = {}
        for node_id in self.iter_topological(roots):
            if node_id in absorbed:
                continue
            node = self._nodes[node_id]
            group = type_to_group.get(node.node_type)
            if group is None:
                continue
            chain_types, absorb_ids = group
            result: list[int] = []
            stack = [node_id]
            while stack:
                current = stack.pop()
                for child in self._nodes[current].children:
                    if child in absorb_ids:
                        continue
                    child_node = self._nodes[child]
                    if child_node.node_type in chain_types and parents[child] == 1:
                        stack.append(child)
                    else:
                        result.append(child)
            flat_children[node_id] = result

        return absorbed, flat_children

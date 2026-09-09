#  Copyright (c) 2024-2026. KU Leuven
"""Circuit class for building logical circuits with configurable operators."""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..algebraic import AlgebraicStructure
from ..algebraic import Semiring
from ..algebraic import get_algebraic_structure
from ..module import DeepLogModule
from ..symbol import Symbol
from ..symbol import is_structure_wrapped
from ..symbol import with_structure
from .graph import Graph
from .graph import Node


if TYPE_CHECKING:
    from ..formula.deeplogformulafactory import DeepLogFormulaFactory


class Circuit:
    """Circuit implementation using an internal graph for efficient construction.

    Conversion to torch modules happens via to_module(), which evaluates the
    circuit as written — in the semiring Klay implements for its structure, or
    node by node through the structure's own ``operator_fns``.
    """

    _counters: dict[str, int] = {}

    def __init__(
        self,
        structure: str | AlgebraicStructure,
    ) -> None:
        """Create a circuit configured for the given structure.

        The circuit is evaluated as written: its operators mean what the
        structure's ``operator_fns`` say they mean.

        Args:
            structure: The algebraic structure (name or AlgebraicStructure instance).
        """
        if isinstance(structure, str):
            self._structure = get_algebraic_structure(structure)
        else:
            self._structure = structure

        name = self._structure.name
        count = Circuit._counters.get(name, 0)
        Circuit._counters[name] = count + 1
        self.name = f"circuit_{name}_{count}"

        # Add "leaf" and constant node types to the graph
        node_types = set(self._structure.operators) | {"leaf", "constant"}
        self._graph = Graph(node_types)

        # Leaves are stored *bare*: the structure tag is redundant with the
        # circuit's own structure, so it is dropped on entry (:meth:`get_leaf_node`)
        # and re-appended on the way out (:meth:`_tagged_leaf_name`). ``_leafs``
        # maps the bare identity to its node id; ``_leaf_names`` is the inverse.
        self._leafs: dict[Symbol, int] = {}
        self._leaf_names: dict[int, Symbol] = {}
        # Constants mirror the leaves: ``_constant_nodes`` maps the bare symbol
        # to its node id, ``_constant_names`` is the inverse, and
        # ``_constant_values`` holds the value the structure resolved it to.
        self._constant_nodes: dict[Symbol, int] = {}
        self._constant_names: dict[int, Symbol] = {}
        self._constant_values: dict[int, float] = {}

    @property
    def structure(self) -> AlgebraicStructure:
        """Return the algebraic structure of this circuit.

        Callers wanting the registry name use ``circuit.structure.name``.
        """
        return self._structure

    @property
    def leaf_nodes(self) -> dict[Symbol, int]:
        """Mapping of leaf symbols to node IDs, keyed by their tagged name.

        Leaves are stored bare; the keys here carry the circuit's structure tag
        re-appended (:meth:`_tagged_leaf_name`), so a predicate leaf appears under
        its wrapped name — the spelling the module boundary matches on.
        """
        return {
            with_structure(bare, self._structure.name): nid
            for nid, bare in self._leaf_names.items()
        }

    def _tagged_leaf_name(self, node_id: int) -> Symbol | None:
        """The structure-tagged display name of a leaf node, or ``None``.

        Leaves are stored bare because the tag is redundant with the circuit's
        own structure (:meth:`get_leaf_node`). This re-appends it, so boundary
        views and the module input shape carry the same wrapped spelling their
        producers (predicate modules, formula atoms) use.
        """
        bare = self._leaf_names.get(node_id)
        return with_structure(bare, self._structure.name) if bare is not None else None

    def get_leaf_name(self, node_id: int) -> Symbol | None:
        """Return the *bare* canonical leaf symbol for a node ID, or None.

        This is the leaf's identity with its redundant structure tag stripped.
        For the structure-tagged boundary spelling use :meth:`get_symbol_name`,
        :attr:`leaf_nodes`, or :meth:`reachable_leaves`.
        """
        return self._leaf_names.get(node_id)

    def get_symbol_name(self, node_id: int) -> Symbol | None:
        """Return the formula symbol represented by a leaf or constant node.

        Both are stored bare and returned with the circuit's tag re-appended,
        so formula-level views expose leaves and constants uniformly as atoms.
        """
        bare = self._leaf_names.get(node_id)
        if bare is None:
            bare = self._constant_names.get(node_id)
        return with_structure(bare, self._structure.name) if bare is not None else None

    def reachable_leaves(
        self, roots: list[int], frontier: Mapping[int, Symbol] | None = None
    ) -> dict[Symbol, int]:
        """The input slots of the subgraph under ``roots``: ``{symbol: node_id}``.

        Circuits are shared (one per structure), so the leaves under a given
        set of roots are typically a strict subset of all leaves ever created.
        Preserves leaf insertion order, keeping the result deterministic. Keyed
        by the structure-tagged name, as the compilers need for input-slot
        indexing; for plain leaf *symbols* use :meth:`reachable_leaf_names`.

        With a ``frontier``, the traversal stops at those nodes and they become
        slots themselves, under the symbols given, after the real leaves in
        traversal order.
        """
        reachable = set(self.iter_topological(roots, frontier))
        slots = {
            with_structure(bare, self._structure.name): nid
            for nid, bare in self._leaf_names.items()
            if nid in reachable
        }
        if frontier:
            slots.update(
                {symbol: nid for nid, symbol in frontier.items() if nid in reachable}
            )
        return slots

    def reachable_constants(
        self, roots: list[int], frontier: Mapping[int, Symbol] | None = None
    ) -> dict[Symbol, int]:
        """The constant nodes reachable from ``roots``: ``{symbol: node_id}``.

        The counterpart to :meth:`reachable_leaves` for the ``constant`` nodes a
        structure's ``constant_fn`` produced — the ``0.6`` in ``times(p, 0.6)``,
        and the identities alongside them, since a constant is one kind of node.
        A backend with a primitive for an identity (:attr:`zero_node`,
        :attr:`one_node`) drops it from this set and binds it instead. Keyed by
        the structure-tagged name, as :meth:`reachable_leaves` is; the value each
        stands for is in :attr:`constant_values`. Topological order, so slot
        assignment is deterministic.
        """
        return {
            with_structure(bare, self._structure.name): node_id
            for node_id in self.iter_topological(roots, frontier)
            if (bare := self._constant_names.get(node_id)) is not None
        }

    def _reachable_named(
        self, roots: list[int], name_of: Callable[[int], Symbol | None]
    ) -> list[Symbol]:
        """Reachable nodes named via ``name_of``, topological order, ``None``s dropped."""
        return [
            name
            for node_id in self.iter_topological(roots)
            if (name := name_of(node_id)) is not None
        ]

    def reachable_leaf_names(self, roots: list[int]) -> list[Symbol]:
        """Named leaves reachable from ``roots``, in topological order (no constants).

        The leaves-only boundary view as plain symbols: what feeds a lump and
        the boolean boundary an expectation transform reasons about. Contrast :meth:`reachable_symbol_names`
        (also yields constants) and :meth:`reachable_leaves` (keys symbols to ids).
        """
        return self._reachable_named(roots, self._tagged_leaf_name)

    def reachable_symbol_names(self, roots: list[int]) -> list[Symbol]:
        """Leaf and constant symbols reachable from ``roots``, in topological order.

        The generic boundary walk for formula-level views
        (:class:`~deeplog.formula.ast.CircuitNode` children): every named leaf
        *and* constant once. Contrast
        :meth:`reachable_leaf_names`, which omits constants.
        """
        return self._reachable_named(roots, self.get_symbol_name)

    def get_leaf_node(self, name: Symbol) -> int:
        """Return a node ID for the given symbol, creating it if needed.

        A structure-wrapped ``name`` (``("_", inner, (structure,))``) must carry
        this circuit's own structure, or :class:`ValueError` is raised. The tag
        is then stripped and the bare inner symbol stored as the canonical leaf
        identity, so both spellings of an atom resolve to one shared node.
        Boundary views re-append the tag (:meth:`_tagged_leaf_name`), so the
        module input shape still matches its predicate modules.
        """
        self._validate_structure_tag(name)
        bare: Symbol = name[1] if is_structure_wrapped(name) else name  # pyright: ignore[reportAssignmentType]

        # A constant of the structure -- an identity or an arbitrary value; the
        # structure resolves both, and the node holds the value it resolved to.
        value = self._structure.get_constant_value(bare)
        if value is not None:
            if bare not in self._constant_nodes:
                node_id = self._graph.add_node("constant", ())
                self._constant_nodes[bare] = node_id
                self._constant_names[node_id] = bare
                self._constant_values[node_id] = value
            return self._constant_nodes[bare]

        if bare not in self._leafs:
            node_id = self._graph.add_leaf()
            self._leafs[bare] = node_id
            self._leaf_names[node_id] = bare
        return self._leafs[bare]

    def _validate_structure_tag(self, name: Symbol) -> None:
        """Raise if ``name`` carries a structure tag that disagrees with the circuit."""
        if not is_structure_wrapped(name):
            return
        tag = name[2][0]
        if tag != self._structure.name:
            raise ValueError(
                f"Leaf symbol structure {tag!r} does not match circuit "
                f"structure {self._structure.name!r}: {name}"
            )

    def get_operator(self, operator: str) -> Callable[..., int]:
        """Return a callable that applies the requested operator.

        Args:
            operator: The operator name (e.g., "and", "or", "not").

        Returns:
            A callable that takes node IDs and returns a new node ID.
        """
        if operator not in self._structure.operators:
            raise ValueError(
                f"Operator '{operator}' is not an operator of "
                f"'{self._structure.name}'. Available: "
                f"{sorted(self._structure.operators)}"
            )

        def apply_operator(*args: int) -> int:
            return self._graph.add_node(operator, tuple(args))

        return apply_operator

    @property
    def constant_nodes(self) -> dict[Symbol, int]:
        """Mapping of constant symbols to node IDs."""
        return self._constant_nodes

    @property
    def constant_values(self) -> dict[int, float]:
        """Mapping of node IDs to their constant float values."""
        return self._constant_values

    @property
    def zero_node(self) -> int | None:
        """Node ID of the additive identity, or ``None`` if not materialized."""
        structure = self._structure
        if not isinstance(structure, Semiring):
            return None
        return self._constant_nodes.get(structure.zero)

    @property
    def one_node(self) -> int | None:
        """Node ID of the multiplicative identity, or ``None`` if not materialized."""
        structure = self._structure
        if not isinstance(structure, Semiring):
            return None
        return self._constant_nodes.get(structure.one)

    def _get_node(self, node_id: int) -> Node:
        """Return the node with the given ID."""
        return self._graph.get_node(node_id)

    def iter_topological(
        self, roots: list[int], frontier: Mapping[int, Symbol] | None = None
    ) -> Iterator[int]:
        """Iterate over nodes in topological order (leaves first).

        ``frontier`` nodes are yielded but not descended into: a compile bounded
        at them treats their values as inputs (:mod:`deeplog.circuit.split`).
        """
        return self._graph.iter_topological(roots, frozenset(frontier or ()))

    def flatten_chains(
        self,
        roots: list[int],
        chain_groups: list[tuple[frozenset[str], frozenset[int]]],
        frontier: Mapping[int, Symbol] | None = None,
    ) -> tuple[set[int], dict[int, list[int]]]:
        """Delegate to :meth:`~deeplog.circuit.graph.Graph.flatten_chains`."""
        return self._graph.flatten_chains(
            roots, chain_groups, frozenset(frontier or ())
        )

    def to_module(
        self,
        roots: dict[int, Symbol],
        frontier: Mapping[int, Symbol] | None = None,
    ) -> DeepLogModule:
        """Convert the circuit to a DeepLog module, evaluating it as written.

        Args:
            roots: A dictionary mapping node IDs to output names.
            frontier: Nodes to lower as input slots under the given
                     symbols rather than descending into
                     (:mod:`deeplog.circuit.split`).

        Returns:
            A DeepLogModule wrapping the circuit as a torch module.
        """
        from .lower.dispatch import to_module

        return to_module(self, roots, frontier=frontier)

    def fold[T](
        self,
        roots: list[int],
        algebra: "DeepLogFormulaFactory[T]",
        *,
        frontier: Mapping[int, Symbol] | None = None,
        memo: dict[int, T] | None = None,
    ) -> dict[int, T]:
        """Re-emit the subgraph under ``roots`` through ``algebra``.

        The "fold" verb on the engine; see
        :func:`~deeplog.circuit.fold.fold_circuit` for the full contract. A
        circuit is a sub-algebra of the formula AST, so this takes the same
        algebra :func:`~deeplog.formula.ast.fold` drives over a tree.
        """
        from .fold import fold_circuit

        return fold_circuit(self, roots, algebra, frontier=frontier, memo=memo)

    def transform(
        self,
        roots: list[int],
        target_structure: str | AlgebraicStructure,
        *,
        operator_mapping: dict[str, str] | None = None,
        leaf_mapping: Callable[[Symbol], Symbol] | None = None,
        into: "tuple[Circuit, dict[int, int]] | None" = None,
    ) -> "tuple[Circuit, dict[int, int]]":
        """Transform the subgraph under ``roots`` into ``target_structure``.

        The "transform" verb on the engine; see
        :func:`~deeplog.circuit.transform.transform_circuit` for the full
        contract. Returns ``(new_circuit, node_map)`` mapping each source node
        id to its id in the new circuit.
        """
        from .transform import transform_circuit

        return transform_circuit(
            self,
            target_structure,
            roots,
            operator_mapping=operator_mapping,
            leaf_mapping=leaf_mapping,
            into=into,
        )

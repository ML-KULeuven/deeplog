#  Copyright (c) 2024-2026. KU Leuven
"""Circuit class for building logical circuits with configurable operators."""

from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass

from ..algebraic import AlgebraicStructure
from ..algebraic import HasStructure
from ..algebraic import Semiring
from ..algebraic import get_algebraic_structure
from ..module import DeepLogModule
from ..module.deeplog_module import SupportsToModule
from ..symbol import Symbol
from ..symbol import is_structure_wrapped
from .graph import Graph
from .graph import Node


class Circuit:
    """Circuit implementation using an internal graph for efficient construction.

    Conversion to torch modules happens via
    to_module(), which can use either Klay or SDD depending on the deterministic flag.
    """

    _counters: dict[str, int] = {}

    def __init__(
        self,
        structure: str | AlgebraicStructure,
        deterministic: bool = False,
    ) -> None:
        """Create a circuit configured for the given structure.

        Args:
            structure: The algebraic structure (name or AlgebraicStructure instance).
            deterministic: If True, use PySDD for deterministic (idempotent) compilation.
                          If False (default), use Klay.
        """
        if isinstance(structure, str):
            self._structure = get_algebraic_structure(structure)
        else:
            self._structure = structure

        self.deterministic = deterministic

        name = self._structure.name
        count = Circuit._counters.get(name, 0)
        Circuit._counters[name] = count + 1
        self.name = f"circuit_{name}_{count}"

        # Add "leaf" and constant node types to the graph
        node_types = (
            set(self._structure.operators)
            | {"leaf", "constant"}
            | set(self._structure.named_constants)
        )
        self._graph = Graph(node_types)

        self._leafs: dict[Symbol, int] = {}
        self._leaf_names: dict[int, Symbol] = {}
        self._constant_nodes: dict[str, int] = {}
        self._constant_values: dict[int, float] = {}

    @property
    def structure(self) -> str:
        """Return the structure name."""
        return self._structure.name

    @property
    def algebraic_structure(self) -> AlgebraicStructure:
        """Return the algebraic structure instance."""
        return self._structure

    @property
    def leaf_nodes(self) -> dict[Symbol, int]:
        """Mapping of leaf symbols to node IDs."""
        return self._leafs

    def get_leaf_name(self, node_id: int) -> Symbol | None:
        """Return the symbol name for a leaf node ID, or None if not found."""
        return self._leaf_names.get(node_id)

    def get_leaf_node(self, name: Symbol) -> int:
        """Return a node ID for the given symbol, creating it if needed.

        When ``name`` is structure-wrapped (``("_", inner, (structure,))``),
        the wrapper's tag must match this circuit's own structure; otherwise
        a :class:`ValueError` is raised. This prevents leaves carrying a
        structure label that disagrees with the circuit they live in.
        """
        self._validate_structure_tag(name)

        # Named constants: bare ("0",) or wrapped ("_", ("0",), (structure,)).
        sym_to_role = self._structure.symbol_to_role
        role = sym_to_role.get(name)
        if role is None and is_structure_wrapped(name) and len(name[1]) == 1:
            role = sym_to_role.get(name[1])  # pyright: ignore[reportArgumentType]
        if role is not None:
            if role not in self._constant_nodes:
                self._constant_nodes[role] = self._graph.add_node(role, ())
            return self._constant_nodes[role]

        # Arbitrary numeric constants resolved via the structure's constant_fn.
        symbol = name[1] if is_structure_wrapped(name) else name
        value = self._structure.get_constant_value(symbol)  # pyright: ignore[reportArgumentType]
        if value is not None:
            key = str(value)
            if key not in self._constant_nodes:
                node_id = self._graph.add_node("constant", ())
                self._constant_nodes[key] = node_id
                self._constant_values[node_id] = value
            return self._constant_nodes[key]

        if name not in self._leafs:
            node_id = self._graph.add_leaf()
            self._leafs[name] = node_id
            self._leaf_names[node_id] = name
        return self._leafs[name]

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
                f"Unknown operator '{operator}'. Available: {self._structure.operators}"
            )

        def apply_operator(*args: int) -> int:
            return self._graph.add_node(operator, tuple(args))

        return apply_operator

    @property
    def constant_nodes(self) -> dict[str, int]:
        """Mapping of constant roles to node IDs."""
        return self._constant_nodes

    @property
    def constant_values(self) -> dict[int, float]:
        """Mapping of node IDs to their constant float values."""
        return self._constant_values

    @property
    def zero_node(self) -> int | None:
        """Node ID of the additive identity, or ``None`` if not materialized."""
        return self._semiring_constant_node(lambda s: s.zero)

    @property
    def one_node(self) -> int | None:
        """Node ID of the multiplicative identity, or ``None`` if not materialized."""
        return self._semiring_constant_node(lambda s: s.one)

    def _semiring_constant_node(self, pick: Callable[[Semiring], Symbol]) -> int | None:
        if not isinstance(self._structure, Semiring):
            return None
        role = self._structure.symbol_to_role.get(pick(self._structure))
        return self._constant_nodes.get(role) if role else None

    def get_node(self, node_id: int) -> Node:
        """Return the node with the given ID."""
        return self._graph.get_node(node_id)

    def iter_topological(self, roots: list[int]) -> Iterator[int]:
        """Iterate over nodes in topological order (leaves first)."""
        return self._graph.iter_topological(roots)

    def flatten_chains(
        self,
        roots: list[int],
        chain_groups: list[tuple[frozenset[str], frozenset[int]]],
    ) -> tuple[set[int], dict[int, list[int]]]:
        """Delegate to :meth:`~deeplog.circuit.graph.Graph.flatten_chains`."""
        return self._graph.flatten_chains(roots, chain_groups)

    def to_module(
        self,
        roots: dict[int, Symbol],
        structure_override: str | None = None,
        categoricals: dict[Symbol, tuple[str, "int | str"]] | None = None,
        labels: dict[Symbol, Symbol] | None = None,
    ) -> DeepLogModule:
        """Convert the circuit to a DeepLog module.

        Args:
            roots: A dictionary mapping node IDs to output names.
            structure_override: If provided, compile using this structure's
                               semiring instead of the circuit's own structure.
            categoricals: For circuits with annotated-disjunction leaves,
                         maps each AD branch leaf goal to ``(cat_id, value)``.
                         When non-empty, compilation uses the MV-SDD backend.
            labels: Optional ``EngineResult.labels`` map. Used by the
                   MV-SDD backend to bake numeric-constant labels into
                   the AC instead of exposing them as runtime input
                   slots.

        Returns:
            A DeepLogModule wrapping the circuit as a torch module.
        """
        from .compile import to_module

        return to_module(
            self,
            roots,
            structure_override=structure_override,
            categoricals=categoricals,
            labels=labels,
        )


@dataclass(unsafe_hash=True)
class CircuitNode(SupportsToModule, HasStructure):
    """A dataclass holding an internal node ID and the circuit it lives in."""

    circuit: Circuit
    node: int

    def to_module(
        self,
        name: Symbol | None = None,
        structure_override: str | None = None,
    ) -> DeepLogModule:
        """Turn the circuit into a module using this node as the root."""
        name = name or (f"{self.circuit.name}_0",)
        return self.circuit.to_module(
            {self.node: name},
            structure_override=structure_override,
        )

    def get_structure(self) -> str:
        """Return the structure of the module."""
        return self.circuit.structure

    def transform_circuit(
        self,
        target_structure: str | AlgebraicStructure,
        *,
        operator_mapping: dict[str, str] | None = None,
        leaf_mapping: Callable[[Symbol], Symbol] | None = None,
        deterministic: bool | None = None,
    ) -> "CircuitNode":
        """Transform this circuit node to a different algebraic structure.

        See :func:`~deeplog.circuit.transform.transform_circuit` for details.
        """
        from .transform import transform_circuit

        new_circuit, node_map = transform_circuit(
            self.circuit,
            target_structure,
            [self.node],
            operator_mapping=operator_mapping,
            leaf_mapping=leaf_mapping,
            deterministic=deterministic,
        )
        return CircuitNode(new_circuit, node_map[self.node])


def transform_nodes(
    *nodes: CircuitNode,
    target_structure: str | AlgebraicStructure,
    operator_mapping: dict[str, str] | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    deterministic: bool | None = None,
) -> tuple[CircuitNode, ...]:
    """Transform one or more CircuitNodes to a different algebraic structure.

    All nodes must be from the same circuit.

    Args:
        *nodes: One or more CircuitNodes to transform.
        target_structure: The target algebraic structure (name or instance).
        operator_mapping: Explicit operator name mapping (see transform_circuit()).
        leaf_mapping: Optional callable to remap leaf symbols.
        deterministic: Deterministic flag for the new circuit.

    Returns:
        A tuple of CircuitNodes in the new circuit, one per input node.
    """
    if not nodes:
        raise ValueError("At least one CircuitNode is required")

    circuit = nodes[0].circuit
    for n in nodes[1:]:
        if n.circuit is not circuit:
            raise ValueError("All CircuitNodes must be from the same circuit")

    from .transform import transform_circuit

    new_circuit, node_map = transform_circuit(
        circuit,
        target_structure,
        [n.node for n in nodes],
        operator_mapping=operator_mapping,
        leaf_mapping=leaf_mapping,
        deterministic=deterministic,
    )
    return tuple(CircuitNode(new_circuit, node_map[n.node]) for n in nodes)


def to_module(
    *nodes: CircuitNode,
    names: tuple[Symbol, ...] | None = None,
    categoricals: dict[Symbol, tuple[str, "int | str"]] | None = None,
    labels: dict[Symbol, Symbol] | None = None,
    structure_override: str | None = None,
) -> DeepLogModule:
    """Convert one or more CircuitNodes into a DeepLogModule.

    All nodes must be from the same circuit.

    Args:
        *nodes: One or more CircuitNodes to use as roots.
        names: Optional output names for each root. If not provided,
               names are auto-generated from the circuit name.
        categoricals: Optional ``EngineResult.categoricals`` map. Pass
                     this when compiling AD-bearing programs to get the
                     MV-SDD backend (canonicalised + mutex-respecting).
        structure_override: Compile with this algebraic structure's
                           semiring instead of the circuit's own
                           (e.g. ``"probability"`` to evaluate the
                           AC as a probability marginal).

    Returns:
        A DeepLogModule with the specified roots.
    """
    if not nodes:
        raise ValueError("At least one CircuitNode is required")

    circuit = nodes[0].circuit
    for n in nodes[1:]:
        if n.circuit is not circuit:
            raise ValueError("All CircuitNodes must be from the same circuit")

    if names is None:
        names = tuple((f"{circuit.name}_{i}",) for i in range(len(nodes)))
    elif len(names) != len(nodes):
        raise ValueError(f"Expected {len(nodes)} names, got {len(names)}")

    roots = {n.node: name for n, name in zip(nodes, names, strict=True)}
    return circuit.to_module(
        roots,
        structure_override=structure_override,
        categoricals=categoricals,
        labels=labels,
    )

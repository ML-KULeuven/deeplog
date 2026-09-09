#  Copyright (c) 2024-2026. KU Leuven
"""Building a decision diagram from a circuit, and reading one back as a circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...algebraic import Semiring
from ...formula.deeplogformulafactory import DeepLogFormulaFactory
from ..backends import circuit_roles
from ..transform import CircuitAlgebra


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ...algebraic import AlgebraicStructure
    from ...formula.ast import CircuitNode
    from ...symbol import Symbol
    from ..circuit import Circuit


class DiagramAlgebra[T](DeepLogFormulaFactory[T]):
    """Build a decision diagram out of a circuit's nodes.

    The circuit fragment of the formula algebra
    (:func:`~deeplog.circuit.fold.fold_circuit`), carried by a diagram node.
    Dispatch is by *role* -- the structure's product becomes ``&``, its sum
    ``|``, its complement ``~`` -- so ``and``/``or``/``not`` and
    ``times``/``plus``/``negate`` are the same three nodes under their two
    spellings. Both knowledge-compilation passes drive it, and their diagram
    nodes expose the bitwise operators.

    ``atoms`` maps each atom the circuit can reach -- every leaf, and the
    identity constants -- to the diagram node standing for it. The caller builds
    it, because only the caller knows the manager the literals belong to.
    """

    def __init__(self, structure: AlgebraicStructure, atoms: dict[Symbol, T]) -> None:
        """Build a diagram over ``structure``'s roles from ``atoms``' literals."""
        roles = circuit_roles(structure)
        self._structure = structure
        self._product = roles.get("product")
        self._sum = roles.get("sum")
        self._negation = roles.get("negation")
        self._atoms = atoms

    def create_atom(self, atom: Symbol) -> T:
        """The diagram node standing for ``atom``."""
        node = self._atoms.get(atom)
        if node is None:
            raise ValueError(
                f"Knowledge compilation has no diagram node for the atom {atom} "
                f"in structure '{self._structure.name}': every leaf and identity "
                f"constant is a literal of the manager, and nothing else is."
            )
        return node

    def create_unary_node(self, operator: str, operand: T) -> T:
        """The complement of ``operand``, if ``operator`` is the complement."""
        if operator != self._negation:
            raise self._no_diagram_node(operator)
        return ~operand  # pyright: ignore[reportOperatorIssue]

    def create_binary_node(self, operator: str, lhs: T, rhs: T) -> T:
        """The conjunction or disjunction of ``lhs`` and ``rhs``, by role."""
        if operator == self._product:
            return lhs & rhs  # pyright: ignore[reportOperatorIssue]
        if operator == self._sum:
            return lhs | rhs  # pyright: ignore[reportOperatorIssue]
        raise self._no_diagram_node(operator)

    def _no_diagram_node(self, operator: str) -> ValueError:
        return ValueError(
            f"Knowledge compilation has no diagram node for "
            f"'{operator}' in structure '{self._structure.name}': it "
            f"builds a diagram out of the product, sum and complement, and a "
            f"structure's other operators have no logical reading."
        )

    # A circuit has no node for the three eliminators below: they are the part
    # of the formula algebra a circuit cannot express.

    def create_transformation(self, structure: str, child: T) -> T:
        """Unreachable: a circuit holds no cross-structure boundary."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; a circuit has no "
            f"transformation, only the structure it was built in."
        )

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[T],
        child: T,
    ) -> T:
        """Unreachable: a circuit holds no aggregation."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; a circuit has no "
            f"aggregation, only the ground formula one was expanded into."
        )

    def embed_circuit(self, node: CircuitNode, children: tuple[T, ...] = ()) -> T:
        """Unreachable: the fold is already inside a circuit."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; it is already inside "
            f"the circuit a lump would embed."
        )


def seed_constants[T](
    circuit: Circuit, atoms: dict[Symbol, T], zero: T, one: T
) -> None:
    """Register the identity constants under the symbols the fold will ask for."""
    for node_id, diagram in ((circuit.zero_node, zero), (circuit.one_node, one)):
        if node_id is None:
            continue
        symbol = circuit.get_symbol_name(node_id)
        if symbol is not None:
            atoms[symbol] = diagram


class DiagramEmitter:
    """Rebuild a compiled diagram as a circuit, through a circuit algebra.

    The other direction of the boundary :class:`DiagramAlgebra` crosses: that
    one is driven *by* a fold, this one drives an algebra itself, because a
    diagram has no uniform shape to fold over -- each backend exposes its own
    node kinds, so each subclass walks its own and calls the eliminators here.

    What it builds with is not its own: a
    :class:`~deeplog.circuit.transform.CircuitAlgebra` writes the nodes, so
    ``conjoin`` / ``disjoin`` / ``negate`` name operators in the *source's*
    spelling and it maps them to the target's. Knowledge compilation preserves
    roles, so the target may be another algebra: its product, sum and complement
    are applied to the same diagram, which is what transforming the emitted
    circuit afterwards would have done. ``leaf_mapping`` renames each atom on the
    way, as that transform would have.

    The :meth:`memoised` key must be the diagram's *canonical* node identity:
    both backends hand out fresh Python wrappers per traversal, so memoising on
    ``id()`` re-emits a shared subdiagram as disjoint copies.
    """

    def __init__(
        self,
        target: Circuit,
        source_structure: AlgebraicStructure,
        leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    ) -> None:
        """Emit into ``target``, naming operators as ``source_structure`` does.

        Raises:
            ValueError: If ``source_structure`` declares no product and sum,
                which a diagram's conjunction and disjunction emit as.
        """
        if not isinstance(source_structure, Semiring):
            raise ValueError(
                f"A diagram emits a product and a sum, which structure "
                f"'{source_structure.name}' does not declare."
            )
        self.target = target
        self._source_structure: Semiring = source_structure
        self._algebra = CircuitAlgebra(
            target, source_structure, leaf_mapping=leaf_mapping
        )
        roles = circuit_roles(source_structure)
        self._product = roles["product"]
        self._sum = roles["sum"]
        self._negation = roles.get("negation")
        self._memo: dict[object, int] = {}

    def leaf(self, symbol: Symbol) -> int:
        """The target node for the atom ``symbol`` names."""
        return self._algebra.create_atom(symbol)

    def zero(self) -> int:
        """The target node for the source's additive identity."""
        return self._algebra.create_atom(self._source_structure.zero)

    def one(self) -> int:
        """The target node for the source's multiplicative identity."""
        return self._algebra.create_atom(self._source_structure.one)

    def conjoin(self, *nodes: int) -> int:
        """A conjunction of ``nodes`` in the target circuit."""
        return self._apply(self._product, nodes)

    def disjoin(self, *nodes: int) -> int:
        """A disjunction of ``nodes`` in the target circuit."""
        return self._apply(self._sum, nodes)

    def negate(self, node: int) -> int:
        """The complement of ``node`` in the target circuit."""
        if self._negation is None:
            raise ValueError(
                f"Structure '{self._source_structure.name}' declares no "
                f"complement, so a negative literal has no circuit form."
            )
        return self._algebra.create_unary_node(self._negation, node)

    def memoised(self, key: object, build) -> int:
        """Return the node for ``key``, building it once."""
        cached = self._memo.get(key)
        if cached is None:
            cached = self._memo[key] = build()
        return cached

    def _apply(self, operator: str, nodes: tuple[int, ...]) -> int:
        """Fold ``nodes`` pairwise: the algebra's eliminator is binary."""
        if not nodes:
            raise ValueError("An operator needs at least one operand.")
        result = nodes[0]
        for node in nodes[1:]:
            result = self._algebra.create_binary_node(operator, result, node)
        return result

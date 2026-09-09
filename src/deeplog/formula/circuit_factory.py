#  Copyright (c) 2024-2026. KU Leuven
"""The circuit-building formula factory.

The *construction* half of lowering: folding an AST through a
:class:`CircuitFactory` rewrites every maximal circuit-representable region into
a :class:`~deeplog.formula.ast.CircuitNode` lump, whose fine
``and``/``or``/``not``/``times`` structure lives in a circuit rather than in
nested AST objects.

A circuit holds only leaves and operator applications, so an atom becomes a leaf
and a unary or binary node an operator node. Anything else either becomes a
*boundary* leaf — one standing for a value the circuit can hold but not compute,
its sub-formula carried in :attr:`~deeplog.formula.ast.CircuitNode.feeders` and
built when the lump is lowered — or stays symbolic. Each eliminator below says
which it does, and every one is *total*: a child that did not collapse leaves
its symbolic node rebuilt over the folded children.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial

from ..algebraic import AlgebraicStructure
from ..circuit import Circuit
from ..symbol import Symbol
from ..symbol import strip_literal_structure
from ..symbol import unwrap_structure
from ..symbol import with_structure
from .ast import Aggregation
from .ast import BinaryOp
from .ast import CircuitNode
from .ast import FormulaNode
from .ast import Transformation
from .ast import UnaryOp
from .circuit_node import lump_name
from .deeplogformulafactory import DeepLogFormulaFactory
from .deeplogmodulefactory.registry import default_circuit_builders
from .strategies import absorb_aggregation


class CircuitFactory(DeepLogFormulaFactory[FormulaNode]):
    """Fold a formula AST into circuit lumps, passing non-circuit nodes through."""

    def __init__(
        self,
        structures: Mapping[str, AlgebraicStructure] | None = None,
    ) -> None:
        """Configure the per-structure circuit builders."""
        self._structures = dict(structures or {})
        self._circuit_builders: dict[str, object] = dict(default_circuit_builders)
        for name, structure in self._structures.items():
            self._circuit_builders[name] = partial(Circuit, structure=structure)
        #: One circuit per structure, the canonical home for every node built
        #: in it; the circuit's own leaf table is the dedup.
        self._circuits: dict[str, Circuit] = {}

    # -- Eliminators (fold calls these bottom-up) ---------------------------

    def create_atom(self, atom: Symbol, structure: str | None = None) -> CircuitNode:
        """Place a leaf for ``atom`` in its structure's circuit.

        ``structure`` is sugar for the tagged-leaf encoding:
        ``create_atom(goal, "boolean")`` is equivalent to
        ``create_atom(("_", goal, ("boolean",)))``.
        """
        if structure is not None:
            atom = ("_", atom, (structure,))
        if len(atom) != 3 or atom[0] != "_":
            raise ValueError(f"Invalid atom: {atom}")
        predicate, arguments = self._decode_leaf(atom)
        circuit = self.get_circuit(predicate[2])
        name: Symbol = ("_", (predicate[0], *arguments), (predicate[2],))
        return self._node(circuit, circuit.get_leaf_node(name))

    def create_unary_node(self, operator: str, operand: FormulaNode) -> FormulaNode:
        """Apply a unary circuit operator, or rebuild the symbolic node."""
        if not isinstance(operand, CircuitNode):
            return UnaryOp(operator, operand)
        circuit = operand.circuit
        return self._node(
            circuit, circuit.get_operator(operator)(operand.node), operand.feeders
        )

    def create_binary_node(
        self, operator: str, lhs: FormulaNode, rhs: FormulaNode
    ) -> FormulaNode:
        """Apply a binary circuit operator, or rebuild the symbolic node.

        Every operator the algebra defines is a circuit node here, a semifield's
        ``divide`` included; whether the compiling backend has a node for it is
        that backend's business (:mod:`deeplog.circuit.split`).
        """
        if not (isinstance(lhs, CircuitNode) and isinstance(rhs, CircuitNode)):
            return BinaryOp(operator, lhs, rhs)
        circuit = lhs.circuit
        if rhs.circuit is not circuit:
            raise ValueError(
                "CircuitFactory cannot combine nodes from different circuits; "
                "a cross-structure boundary must go through create_transformation."
            )
        return self._node(
            circuit,
            circuit.get_operator(operator)(lhs.node, rhs.node),
            self._merge_feeders(lhs.feeders, rhs.feeders),
        )

    def create_transformation(self, structure: str, child: FormulaNode) -> FormulaNode:
        """Cast ``child`` into ``structure``, deferring the elementwise cast.

        A self-describing ``transform`` leaf is placed in the target circuit and
        ``child`` recorded as that leaf's symbolic boundary feeder
        (:attr:`~deeplog.formula.ast.CircuitNode.feeders`). The cast module is
        built only when the lump is lowered, so co-resident casts share one
        source compilation.
        """
        if not isinstance(child, CircuitNode):
            return Transformation(structure, child)
        target = self.get_circuit(structure)
        # ``child_name`` must be *exactly* the output symbol the lowered child
        # lump will carry, because the cast spine built over that lump is wired
        # onto this leaf by symbol equality. The lump's root is labelled with
        # its own circuit's algebra (``circuit.lower.to_module``), so the name
        # embedded here is labelled the same way. Let the two drift apart and
        # nothing raises: the leaf silently becomes an external input and the
        # spine's output is computed and discarded.
        child_name = with_structure(lump_name(child), child.circuit.structure.name)
        leaf_name = with_structure(("transform", (structure,), child_name), structure)
        leaf_id = target.get_leaf_node(leaf_name)
        feeder = Transformation(structure, child)
        return self._node(target, leaf_id, ((leaf_name, feeder),))

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[FormulaNode],
        child: FormulaNode,
    ) -> FormulaNode:
        """Absorb a circuit-representable aggregation, or rebuild the symbolic node.

        An aggregation binds variables, which no circuit can express. *Which*
        aggregations nonetheless have a circuit form is
        :func:`~deeplog.formula.strategies.absorb_aggregation`'s to decide, not
        this fold's; it returns ``None`` for the ones that do not.
        """
        lump = absorb_aggregation(self.get_circuit, operation, binders, params, child)
        if lump is not None:
            return lump
        return Aggregation(operation, tuple(binders), tuple(params), child)

    def embed_circuit(
        self, node: CircuitNode, children: tuple[FormulaNode, ...] = ()
    ) -> CircuitNode:
        """A compiled lump embeds into the circuit world as itself.

        :attr:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.lowers_circuit_children`
        is left off, so the fold never descends a lump's boundary here and
        ``children`` is always empty.
        """
        return node

    def get_circuit(self, structure: str) -> Circuit:
        """Get or create the shared source circuit for a structure."""
        if structure not in self._circuits:
            self._circuits[structure] = self._circuit_builders[structure]()  # type: ignore[operator]
        return self._circuits[structure]

    @staticmethod
    def _node(
        circuit: Circuit,
        node_id: int,
        feeders: tuple[tuple[Symbol, FormulaNode], ...] = (),
    ) -> CircuitNode:
        """A raw AST :class:`~deeplog.formula.ast.CircuitNode` over ``circuit``."""
        return CircuitNode(circuit, node_id, feeders)

    @staticmethod
    def _merge_feeders(
        *feeder_tuples: tuple[tuple[Symbol, FormulaNode], ...],
    ) -> tuple[tuple[Symbol, FormulaNode], ...]:
        """Union symbolic boundary feeders, deduped by leaf symbol."""
        merged: dict[Symbol, FormulaNode] = {}
        for feeders in feeder_tuples:
            merged.update(feeders)
        return tuple(merged.items())

    @staticmethod
    def _decode_leaf(atom: Symbol) -> tuple[tuple[str, int, str], tuple[Symbol, ...]]:
        """Split a leaf atom into its ``(functor, arity, structure)`` key and args."""
        structure_tag = atom[2]
        structure = structure_tag[0]
        literal = strip_literal_structure(unwrap_structure(atom), structure)
        return (literal[0], len(literal) - 1, structure), literal[1:]

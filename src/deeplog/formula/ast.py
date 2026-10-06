#  Copyright (c) 2024-2026. KU Leuven
"""The materialized formula AST and its fold (catamorphism).

DeepLog formulas are parsed into this immutable tree of dataclasses. The symbolic
node kinds mirror the eliminators of
:class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`: the AST is
the *free term* over that signature, and :func:`fold` is the unique catamorphism
that re-emits a term through any factory — so the existing factories
(``SymbolicFormulaFactory`` → text, ``CircuitFactory`` → circuit lumps) become
*interpreters* of the AST. A :class:`~deeplog.formula.ast.CircuitNode` may also
appear as a compressed AST node whose unary/binary structure is stored in a
circuit graph instead of as nested Python objects; ``fold`` splices it in via
``embed_circuit``.
:func:`map_children` is the shallow functor map — the companion to :func:`fold` —
that rebuilds a node from rewritten children.
Every node kind renders itself through :class:`FormulaNodeDisplay`: ``str`` for the
formula's surface syntax, ``repr`` and :meth:`FormulaNodeDisplay.tree` for the node
structure that syntax parses into.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import fields
from functools import cached_property
from typing import TYPE_CHECKING
from typing import cast

from ..algebraic import PROBABILITY
from ..symbol import Symbol
from ..symbol import is_variable
from ..symbol import retag
from ..symbol import structure_of
from ..symbol import symbol_to_pretty_string
from ..symbol import symbol_to_str
from ..symbol import without_structure
from .deeplogformulafactory import DeepLogFormulaFactory
from .symbolic_factory import SymbolicFormulaFactory


if TYPE_CHECKING:
    from ..circuit.circuit import Circuit


class FormulaNodeDisplay:
    """The renderings every :data:`FormulaNode` kind shares.

    ``str`` gives the node's surface syntax, ``repr`` its structure on one line,
    and :meth:`tree` that same structure indented over one line per node.
    """

    def __str__(self) -> str:
        """Render this node as formula text, in the surface syntax of the parser."""
        return fold(cast("FormulaNode", self), _DISPLAY)

    def __repr__(self) -> str:
        """Render this node's structure on one line, as ``Kind(datum, children...)``."""
        return _compact(cast("FormulaNode", self))

    def tree(self) -> str:
        """Render this node's structure as an indented tree, one line per node."""
        return _tree(cast("FormulaNode", self))


class _ValueHashed:
    """A frozen node hashed by value, once, when it is built.

    Its children are built before it and hash in constant time, so hashing a
    formula does not recurse however deeply it nests. It pickles as its fields,
    so a loaded node hashes afresh.
    """

    _hash: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((type(self), *self._values())))

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(self):
        return type(self), self._values()

    def _values(self) -> tuple:
        return tuple(getattr(self, field.name) for field in fields(self))  # pyright: ignore[reportArgumentType]


@dataclass(frozen=True, repr=False)
class Atom(_ValueHashed, FormulaNodeDisplay):
    """A leaf atom.

    ``atom`` is the structure-wrapped symbol ``("_", inner, (structure,))``,
    exactly as
    :meth:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.create_atom`
    receives it. ``=`` is symmetric, and an ``=`` comparing a written value with
    a variable holds the variable first: ``Atom`` stores ``=(a, X)`` as
    ``=(X, a)``, so the two spellings are one atom.
    """

    atom: Symbol
    __hash__ = _ValueHashed.__hash__

    def __post_init__(self) -> None:
        """Hold an ``=`` of a written value and a variable with the variable first."""
        object.__setattr__(self, "atom", _variable_first(self.atom))
        super().__post_init__()

    @property
    def structure(self) -> str | None:
        """The algebraic structure wrapping this atom, or ``None`` if it is bare."""
        return structure_of(self.atom)


def _variable_first(atom: Symbol) -> Symbol:
    """``atom``, with the variable first if it is ``=`` of a written value and a variable."""
    inner = without_structure(atom)
    if (
        len(inner) != 3
        or inner[0] != "="
        or is_variable(inner[1])
        or not is_variable(inner[2])
    ):
        return atom
    return retag(("=", inner[2], inner[1]), atom)


@dataclass(frozen=True, repr=False)
class UnaryOp(_ValueHashed, FormulaNodeDisplay):
    """A unary operator applied to a single operand."""

    operator: str
    operand: FormulaNode
    __hash__ = _ValueHashed.__hash__

    @property
    def structure(self) -> str | None:
        """A unary operator carries its operand's structure."""
        return self.operand.structure


@dataclass(frozen=True, repr=False)
class BinaryOp(_ValueHashed, FormulaNodeDisplay):
    """A binary operator combining two operands."""

    operator: str
    lhs: FormulaNode
    rhs: FormulaNode
    __hash__ = _ValueHashed.__hash__

    @property
    def structure(self) -> str | None:
        """The structure shared by both operands, or ``None`` if they disagree."""
        left, right = self.lhs.structure, self.rhs.structure
        return left if left == right else None


@dataclass(frozen=True, repr=False)
class Transformation(_ValueHashed, FormulaNodeDisplay):
    """A structure-conversion node mapping ``child`` into ``structure``."""

    structure: str
    child: FormulaNode
    __hash__ = _ValueHashed.__hash__


@dataclass(frozen=True, repr=False)
class Aggregation(_ValueHashed, FormulaNodeDisplay):
    """An aggregation over ``binders`` applying ``operation`` to ``child``.

    Unless ``operation`` is ``expectation``, its value is its algebra's
    aggregator for ``operation``
    (:attr:`~deeplog.algebraic.AlgebraicStructure.aggregation_fns`) of
    ``child`` and ``params`` over every assignment of ``binders``.

    Each binder is a variable. ``binders`` and ``params`` are tuples (not lists)
    so the node stays hashable and immutable; :func:`fold` converts ``binders``
    back to a list at the factory call boundary.

    Raises:
        ValueError: If a binder is not a variable.
    """

    operation: str
    binders: tuple[Symbol, ...]
    params: tuple[FormulaNode, ...]
    child: FormulaNode
    __hash__ = _ValueHashed.__hash__

    def __post_init__(self) -> None:
        """Refuse a binder that is not a variable, then hash the node."""
        for binder in self.binders:
            if not is_variable(binder):
                raise ValueError(
                    f"{self.operation} binds {symbol_to_pretty_string(binder)}, "
                    f"which is not a variable; a variable starts with an uppercase "
                    f"letter or an underscore."
                )
        super().__post_init__()

    @property
    def structure(self) -> str | None:
        """An ``expectation`` is a value of its distribution's algebra, or of
        probability without one; any other aggregation carries its child's
        structure."""
        if self.operation == "expectation":
            return self.params[0].structure if self.params else PROBABILITY.name
        return self.child.structure


@dataclass(frozen=True, eq=False, repr=False)
class CircuitNode(FormulaNodeDisplay):
    """A graph-backed formula node — a handle into a circuit graph.

    Pairs a ``circuit`` with the ``node`` id of one of its roots; this is the
    compressed counterpart to symbolic :class:`UnaryOp` / :class:`BinaryOp`
    structure. It lives in the AST (a member of :data:`FormulaNode`) but its fine
    internal structure — the boolean ``and``/``or``/``not`` graph — lives in the
    circuit, not as nested dataclass objects.

    The compressed chunk has a *boundary*: the leaf / constant symbols reachable
    from ``node``. Each boundary symbol feeds a child formula, and ``feeders``
    records the non-trivial ones as a ``name → child`` mapping. A symbol absent
    from ``feeders`` feeds itself as ``Atom(name)`` — so the default empty
    ``feeders`` is the *identity* boundary (every leaf feeds its own atom),
    reproducing the plain leaf view. A non-identity entry makes the boundary
    *symbolic*: a leaf can be fed by an arbitrary sub-formula — including one in a
    different algebraic structure — without re-materializing the circuit.

    Pure, frozen, value-keyed data like its symbolic siblings: the boundary map
    is part of value identity (``feeders`` is a hashable tuple of pairs), so two
    lumps over the same circuit node with different feeders are distinct nodes.
    """

    circuit: Circuit
    node: int
    feeders: tuple[tuple[Symbol, FormulaNode], ...] = ()

    def __eq__(self, other: object) -> bool:
        """Value identity is ``(circuit, node, feeders)``, including subclasses."""
        return (
            isinstance(other, CircuitNode)
            and self.circuit is other.circuit
            and self.node == other.node
            and self.feeders == other.feeders
        )

    def __hash__(self) -> int:
        """Hash by the same identity used for equality."""
        return hash((id(self.circuit), self.node, self.feeders))

    @property
    def structure(self) -> str:
        """The algebraic structure name of the backing circuit."""
        return self.circuit.structure.name

    @cached_property
    def children(self) -> tuple[FormulaNode, ...]:
        """Boundary children of this compressed AST chunk.

        A ``CircuitNode`` compresses a whole circuit subgraph, so its formula
        children are what feeds that subgraph's boundary, in :attr:`boundary`'s
        order, not the direct graph-node successors.
        """
        return tuple(child for _, child in self.boundary)

    @cached_property
    def boundary(self) -> tuple[tuple[Symbol, FormulaNode], ...]:
        """Each leaf / constant symbol reachable from ``node``, with the child feeding it.

        The symbols come in the circuit's deterministic topological order, each
        fed by its ``feeders`` override or, by default, its identity ``Atom``,
        which may name it otherwise: an ``Atom`` holds ``=`` with the variable
        first.

        Cached: computing it walks the whole reachable subgraph and allocates an
        ``Atom`` per boundary symbol, yet it is read by every walk that reaches
        the lump through :func:`children`, the lowering's among them. Caching also
        keeps the identity of those ``Atom``s stable for as long as the lump
        lives, which the lowering's identity-keyed bookkeeping relies on.
        Reachability from a fixed node never changes — a circuit only ever grows
        new nodes — so the cache cannot go stale. (``cached_property`` writes straight into
        ``__dict__``, so it works on this frozen dataclass, and neither ``__eq__``
        nor ``__hash__`` consults it.)
        """
        overrides = dict(self.feeders)
        return tuple(
            (name, overrides.get(name, Atom(name)))
            for name in self.circuit.reachable_symbol_names([self.node])
        )


#: A formula node is either still-*symbolic* (``Atom``/``UnaryOp``/``BinaryOp``/
#: ``Transformation``/``Aggregation``) or a graph-backed
#: :class:`~deeplog.formula.ast.CircuitNode`,
#: whose fine internal structure lives in the circuit rather than nested objects.
#:
#: Every node exposes a ``.structure`` property (``str | None``) — the
#: symbolic-layer counterpart to the module layer's
#: :func:`~deeplog.shape.sole_structure` over a module's output symbols. Both
#: read a label off data rather than off an object — an AST node carries it on
#: its atom, a module on the symbols naming its outputs.
FormulaNode = Atom | UnaryOp | BinaryOp | Transformation | Aggregation | CircuitNode


def fold[T](
    node: FormulaNode,
    factory: DeepLogFormulaFactory[T],
    *,
    memo: dict[int, tuple[FormulaNode, T]] | None = None,
) -> T:
    """Re-emit ``node`` through ``factory`` — the unique catamorphism.

    Visits children before parents (params then child; lhs then rhs) in the same
    order the parser produced them, so folding through a stateful factory such as
    :class:`~deeplog.formula.circuit_factory.CircuitFactory` reproduces the
    original ``create_*`` call sequence (and hence its behaviour).

    The AST is a canonical DAG (:func:`hash_cons`), so structurally-equal
    subformulas are the *same object*. A per-call memo (keyed by node identity)
    folds each shared node once and reuses its carrier: for a circuit factory the
    reused carrier is a shared :class:`~deeplog.formula.ast.CircuitNode` handle,
    so a textually-shared subformula compiles to a single shared circuit node
    (linear, not duplicated) - exploiting the DAG rather than merely representing
    it. The memo is value-preserving (the catamorphism is pure over the DAG).
    ``memo`` seeds it and is mutated in place, as
    :func:`~deeplog.circuit.fold.fold_circuit`'s does, so successive folds through
    one factory build a shared subformula once between them. It maps a node's
    ``id`` to the node and its result: holding the node keeps its ``id`` from
    being reused by another node while the memo lives.

    A :class:`~deeplog.formula.ast.CircuitNode` is a compiled region, so it
    reaches :meth:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.embed_circuit`
    as it is: the fold does not descend its boundary.
    """
    if memo is None:
        memo = {}
    # An explicit stack rather than recursion: a grounded proof can be a chain of
    # disjunctions thousands deep.
    stack: list[tuple[FormulaNode, bool]] = [(node, False)]
    while stack:
        current, ready = stack.pop()
        if id(current) in memo:
            continue
        if not ready:
            stack.append((current, True))
            stack.extend((operand, False) for operand in reversed(operands(current)))
            continue
        memo[id(current)] = (current, _eliminate(current, factory, memo))
    return memo[id(node)][1]


def operands(node: FormulaNode) -> tuple[FormulaNode, ...]:
    """The nodes ``node`` is built from, in the order :func:`fold` folds them.

    A lump's boundary is not among them: a lump is compiled, not folded.
    """
    # Tests in order of frequency: walks call this once per node of a formula.
    if isinstance(node, BinaryOp):
        return (node.lhs, node.rhs)
    if isinstance(node, Atom | CircuitNode):
        return ()
    if isinstance(node, UnaryOp):
        return (node.operand,)
    if isinstance(node, Transformation):
        return (node.child,)
    if isinstance(node, Aggregation):
        return (*node.params, node.child)
    raise TypeError(f"Unknown formula node: {node!r}")


def _eliminate[T](
    node: FormulaNode,
    factory: DeepLogFormulaFactory[T],
    memo: dict[int, tuple[FormulaNode, T]],
) -> T:
    """Call ``factory``'s eliminator for ``node`` on its operands' results in ``memo``."""

    def result(operand: FormulaNode) -> T:
        return memo[id(operand)][1]

    match node:
        case Atom(atom):
            return factory.create_atom(atom)
        case UnaryOp(operator, operand):
            return factory.create_unary_node(operator, result(operand))
        case BinaryOp(operator, lhs, rhs):
            return factory.create_binary_node(operator, result(lhs), result(rhs))
        case Transformation(structure, child):
            return factory.create_transformation(structure, result(child))
        case Aggregation(operation, binders, params, child):
            return factory.create_aggregation(
                operation, list(binders), [result(p) for p in params], result(child)
            )
        case CircuitNode():
            return factory.embed_circuit(node)
    raise TypeError(f"Unknown formula node: {node!r}")


def children(node: FormulaNode) -> tuple[FormulaNode, ...]:
    """The immediate children of ``node``, in fold order.

    The shallow projection companion to :func:`map_children`: that rebuilds a
    node from mapped children, this only reads them, which is what a walk that
    rewrites nothing needs. A :class:`CircuitNode`'s children are its *boundary*
    (:attr:`CircuitNode.children`) rather than its graph successors, so a walk
    sees the computations a lump defers, not its compiled interior.
    """
    match node:
        case Atom():
            return ()
        case UnaryOp(_, operand):
            return (operand,)
        case BinaryOp(_, lhs, rhs):
            return (lhs, rhs)
        case Transformation(_, child):
            return (child,)
        case Aggregation(_, _, params, child):
            return (*params, child)
        case CircuitNode():
            return node.children
        case _:
            raise TypeError(f"Unknown formula node: {node!r}")


def map_children(
    node: FormulaNode, f: Callable[[FormulaNode], FormulaNode]
) -> FormulaNode:
    """Rebuild ``node`` with ``f`` applied to each immediate child.

    The shallow functor map, companion to the recursive :func:`fold`. Where
    ``fold`` collapses the whole tree into a foreign carrier ``T``,
    ``map_children`` stays within the AST and descends exactly one level, leaving
    the recursion to ``f``. It is the building block for AST→AST rewrite passes: a
    bottom-up pass is an ``f`` that calls ``map_children(node, f)`` first and then
    rewrites ``node`` itself (see :func:`hash_cons`).
    """
    match node:
        case Atom():
            return node
        case CircuitNode():
            overrides = dict(node.feeders)
            rebuilt = tuple(
                (name, child)
                for name in node.circuit.reachable_symbol_names([node.node])
                if (child := f(overrides.get(name, Atom(name)))) != Atom(name)
            )
            if rebuilt == node.feeders:
                return node
            return CircuitNode(node.circuit, node.node, rebuilt)
        case UnaryOp(operator, operand):
            return UnaryOp(operator, f(operand))
        case BinaryOp(operator, lhs, rhs):
            return BinaryOp(operator, f(lhs), f(rhs))
        case Transformation(structure, child):
            return Transformation(structure, f(child))
        case Aggregation(operation, binders, params, child):
            return Aggregation(
                operation, binders, tuple(f(p) for p in params), f(child)
            )
    raise TypeError(f"Unknown formula node: {node!r}")


def hash_cons(node: FormulaNode, table: dict[FormulaNode, FormulaNode]) -> FormulaNode:
    """Intern ``node`` and its subtree into ``table``, returning the canonical DAG.

    Bottom-up: children are interned first (via :func:`map_children`), so two
    structurally-equal subformulas become the *same object* — the parsed tree
    collapses to a canonical DAG keyed by value. ``table`` is the intern map,
    threaded by the caller: a parse interns one formula into a fresh table, and
    :meth:`~deeplog.formula.lowering.compiler.Compiler.compile` all of its
    formulas into one. Leaves bottom out for free: ``map_children`` returns an
    ``Atom``/``CircuitNode`` unchanged, and ``setdefault`` interns it by value.

    Purely representational — interning never changes a node's *value*, so
    structural ``==`` is preserved and :func:`fold` re-emits the identical
    ``create_*`` stream — one ``create_*`` call per interned node, since its memo
    keys on the identity interning just made canonical.
    """
    interned: dict[int, FormulaNode] = {}

    def canonical(child: FormulaNode) -> FormulaNode:
        # A lump's unfed leaf is a fresh Atom each time map_children builds it.
        return interned[id(child)] if id(child) in interned else hash_cons(child, table)

    # An explicit stack rather than recursion, as in fold: parsed text can nest
    # thousands deep.
    stack: list[tuple[FormulaNode, bool]] = [(node, False)]
    while stack:
        current, ready = stack.pop()
        if id(current) in interned:
            continue
        if not ready:
            stack.append((current, True))
            stack.extend((child, False) for child in reversed(children(current)))
            continue
        rebuilt = map_children(current, canonical)
        interned[id(current)] = table.setdefault(rebuilt, rebuilt)
    return interned[id(node)]


def _datum(node: FormulaNode) -> str:
    """What ``node`` is beyond its kind: its atom, operator, structure or handle."""
    match node:
        case Atom(atom):
            return _DISPLAY.create_atom(atom)
        case UnaryOp(operator, _) | BinaryOp(operator, _, _):
            return operator
        case Transformation(structure, _):
            return structure
        case Aggregation(operation, binders, _, _):
            return f"{operation}({', '.join(symbol_to_str(b) for b in binders)})"
        case CircuitNode():
            return f"{node.circuit.name}#{node.node}"
        case _:
            raise TypeError(f"Unknown formula node: {type(node).__name__}")


def _compact(node: FormulaNode) -> str:
    """Render ``node`` on one line as ``Kind(datum, children...)``."""
    parts = [_datum(node), *(_compact(child) for child in children(node))]
    return f"{type(node).__name__}({', '.join(parts)})"


def _tree(node: FormulaNode) -> str:
    """Render ``node`` as an indented tree, one ``Kind datum`` line per node."""
    lines = [f"{type(node).__name__} {_datum(node)}"]
    kids = children(node)
    for index, child in enumerate(kids):
        last = index == len(kids) - 1
        head, *rest = _tree(child).split("\n")
        lines.append(("└─ " if last else "├─ ") + head)
        lines.extend(("   " if last else "│  ") + line for line in rest)
    return "\n".join(lines)


class _DisplayFactory(SymbolicFormulaFactory):
    """Formula text in which a compiled lump is spelled as its circuit handle."""

    def embed_circuit(self, node: CircuitNode) -> str:
        """Render a lump as its handle: its interior is compiled, not text."""
        return _datum(node)


_DISPLAY = _DisplayFactory()

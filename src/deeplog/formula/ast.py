#  Copyright (c) 2024-2026. KU Leuven
"""The materialized formula AST and its fold (catamorphism).

DeepLog formulas are parsed into this immutable tree of dataclasses. The symbolic
node kinds mirror the eliminators of
:class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`: the AST is
the *free term* over that signature, and :func:`fold` is the unique catamorphism
that re-emits a term through any factory — so the existing factories
(``SymbolicFormulaFactory`` → text, ``DeepLogModuleFactory`` → modules) become
*interpreters* of the AST. A :class:`~deeplog.formula.ast.CircuitNode` may also
appear as a compressed AST node whose unary/binary structure is stored in a
circuit graph instead of as nested Python objects; ``fold`` splices it in via
``embed_circuit``.
Rewrite passes (e.g. expectation recognition) operate on the symbolic tree before
it is folded to a module; :func:`map_children` is the shallow functor map — the
companion to :func:`fold` — that those passes recurse with.
Every node kind renders itself through :class:`FormulaNodeDisplay`: ``str`` for the
formula's surface syntax, ``repr`` and :meth:`FormulaNodeDisplay.tree` for the node
structure that syntax parses into.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from ..algebraic import PROBABILITY
from ..circuit.circuit import Circuit
from ..symbol import Symbol
from ..symbol import structure_of
from ..symbol import symbol_to_str
from .deeplogformulafactory import DeepLogFormulaFactory
from .symbolic_factory import SymbolicFormulaFactory


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


@dataclass(frozen=True, repr=False)
class Atom(FormulaNodeDisplay):
    """A leaf atom.

    ``atom`` is the structure-wrapped symbol ``("_", inner, (structure,))``,
    exactly as
    :meth:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.create_atom`
    receives it.
    """

    atom: Symbol

    @property
    def structure(self) -> str | None:
        """The algebraic structure wrapping this atom, or ``None`` if it is bare."""
        return structure_of(self.atom)


@dataclass(frozen=True, repr=False)
class UnaryOp(FormulaNodeDisplay):
    """A unary operator applied to a single operand."""

    operator: str
    operand: FormulaNode

    @property
    def structure(self) -> str | None:
        """A unary operator carries its operand's structure."""
        return self.operand.structure


@dataclass(frozen=True, repr=False)
class BinaryOp(FormulaNodeDisplay):
    """A binary operator combining two operands."""

    operator: str
    lhs: FormulaNode
    rhs: FormulaNode

    @property
    def structure(self) -> str | None:
        """The structure shared by both operands, or ``None`` if they disagree."""
        left, right = self.lhs.structure, self.rhs.structure
        return left if left == right else None


@dataclass(frozen=True, repr=False)
class Transformation(FormulaNodeDisplay):
    """A structure-conversion node mapping ``child`` into ``structure``."""

    structure: str
    child: FormulaNode


@dataclass(frozen=True, repr=False)
class Aggregation(FormulaNodeDisplay):
    """An aggregation over ``binders`` applying ``operation`` to ``child``.

    ``binders`` and ``params`` are tuples (not lists) so the node stays hashable
    and immutable; :func:`fold` converts ``binders`` back to a list at the factory
    call boundary.
    """

    operation: str
    binders: tuple[Symbol, ...]
    params: tuple[FormulaNode, ...]
    child: FormulaNode

    @property
    def structure(self) -> str | None:
        """An ``expectation`` yields probability; any other aggregation carries
        its child's structure."""
        if self.operation == "expectation":
            return PROBABILITY.name
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
        children are what feeds that subgraph's boundary: one per leaf / constant
        symbol reachable from ``node`` (in the circuit's deterministic topological
        order), each the symbol's ``feeders`` override or, by default, its
        identity ``Atom`` — not the direct graph-node successors.

        Cached: computing it walks the whole reachable subgraph and allocates an
        ``Atom`` per boundary symbol, yet it is read once per AST pass per lump
        (:func:`fold`, :func:`map_children`, and hence every rewrite pass).
        Caching also keeps the identity of those ``Atom``s stable for as long as
        the lump lives, which the identity-keyed fold memo relies on. Reachability
        from a fixed node never changes — a circuit only ever grows new nodes —
        so the cache cannot go stale. (``cached_property`` writes straight into
        ``__dict__``, so it works on this frozen dataclass, and neither ``__eq__``
        nor ``__hash__`` consults it.)
        """
        overrides = dict(self.feeders)
        return tuple(
            overrides.get(name, Atom(name))
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
    memo: dict[int, T] | None = None,
) -> T:
    """Re-emit ``node`` through ``factory`` — the unique catamorphism.

    Visits children before parents (params then child; lhs then rhs) in the same
    order the parser produced them, so folding through a stateful factory such as
    :class:`DeepLogModuleFactory` reproduces the original ``create_*`` call
    sequence (and hence its behaviour).

    The AST is a canonical DAG (:func:`hash_cons`), so structurally-equal
    subformulas are the *same object*. A per-call memo (keyed by node identity)
    folds each shared node once and reuses its carrier: for a circuit factory the
    reused carrier is a shared :class:`~deeplog.formula.ast.CircuitNode` handle,
    so a textually-shared subformula compiles to a single shared circuit node
    (linear, not duplicated) - exploiting the DAG rather than merely representing
    it. The memo is value-preserving (the catamorphism is pure over the DAG).
    ``memo`` seeds it and is mutated in place, as
    :func:`~deeplog.circuit.fold.fold_circuit`'s does, so successive folds through
    one factory build a shared subformula once between them -- which is how a
    multi-root lowering shares its subformulas, its circuits and its predicate
    modules.

    A :class:`~deeplog.formula.ast.CircuitNode` is graph-backed, but its
    *boundary* (the leaf / cast children, see
    :attr:`~deeplog.formula.ast.CircuitNode.children`) is a derived AST view. A
    factory that sets
    :attr:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.lowers_circuit_children`
    has that boundary folded through the ordinary eliminators (``create_atom``
    per leaf, ``create_transformation`` per cast) and the results handed to
    :meth:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.embed_circuit`,
    so a lump is transparent to the fold rather than an opaque leaf. Factories
    that leave the flag off (the circuit builder, the text interpreter) get the
    node spliced in verbatim with no children. Either way the *interior* boolean
    graph stays compiled in the circuit - only the boundary is re-materialized
    as AST.
    """
    if memo is None:
        memo = {}
    if id(node) in memo:
        return memo[id(node)]
    match node:
        case Atom(atom):
            result = factory.create_atom(atom)
        case UnaryOp(operator, operand):
            result = factory.create_unary_node(
                operator, fold(operand, factory, memo=memo)
            )
        case BinaryOp(operator, lhs, rhs):
            result = factory.create_binary_node(
                operator, fold(lhs, factory, memo=memo), fold(rhs, factory, memo=memo)
            )
        case Transformation(structure, child):
            result = factory.create_transformation(
                structure, fold(child, factory, memo=memo)
            )
        case Aggregation(operation, binders, params, child):
            result = factory.create_aggregation(
                operation,
                list(binders),
                [fold(p, factory, memo=memo) for p in params],
                fold(child, factory, memo=memo),
            )
        case CircuitNode():
            children = (
                tuple(fold(child, factory, memo=memo) for child in node.children)
                if factory.lowers_circuit_children
                else ()
            )
            result = factory.embed_circuit(node, children)
        case _:
            raise TypeError(f"Unknown formula node: {node!r}")
    memo[id(node)] = result
    return result


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
    rewrites ``node`` itself (see
    :func:`deeplog.formula.passes.recognize_expectation`).
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
    threaded by the caller so each parse gets a fresh, self-contained namespace
    (no global leak). Leaves bottom out for free: ``map_children`` returns an
    ``Atom``/``CircuitNode`` unchanged, and ``setdefault`` interns it by value.

    Purely representational — interning never changes a node's *value*, so
    structural ``==`` is preserved and :func:`fold` re-emits the identical
    ``create_*`` stream — one ``create_*`` call per interned node, since its memo
    keys on the identity interning just made canonical.
    """
    node = map_children(node, lambda c: hash_cons(c, table))
    return table.setdefault(node, node)


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

    def embed_circuit(self, node: CircuitNode, children: tuple[str, ...] = ()) -> str:
        """Render a lump as its handle: its interior is compiled, not text."""
        return _datum(node)


_DISPLAY = _DisplayFactory()

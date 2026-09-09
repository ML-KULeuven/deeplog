#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the materialized formula AST — the shape the parser produces.

The fold (catamorphism) is exercised end-to-end elsewhere: ``parse_formula`` is
``fold ∘ parse_formula_to_ast``, so the round-trip suites in ``test_text_parser``
and ``test_text_printer`` already drive it, and ``test_recognition`` covers the
passes/module path. This module therefore pins only what is unique to the IR:
that ``parse_formula_to_ast`` materializes the expected, immutable dataclasses.
"""

import pytest

from deeplog import CircuitNode
from deeplog import DeepLogModule
from deeplog.formula import Aggregation
from deeplog.formula import AstFactory
from deeplog.formula import Atom
from deeplog.formula import BinaryOp
from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula import Transformation
from deeplog.formula import UnaryOp
from deeplog.formula import fold
from deeplog.formula import map_children
from deeplog.formula import parse_formula_to_ast
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory
from deeplog.formula.deeplogmodulefactory import DeepLogModuleFactory


def test_parse_formula_to_ast_aggregation_with_params():
    ast = parse_formula_to_ast("expect(X; q(X)_probability): p(X)_probability")
    assert ast == Aggregation(
        operation="expect",
        binders=(("X",),),
        params=(Atom(("_", ("q", ("X",)), ("probability",))),),
        child=Atom(("_", ("p", ("X",)), ("probability",))),
    )


def test_parse_formula_to_ast_binary_and_transformation():
    ast = parse_formula_to_ast("(=(X,true)_boolean or =(Y,true)_boolean)_probability")
    assert ast == Transformation(
        "probability",
        BinaryOp(
            "or",
            Atom(("_", ("=", ("X",), ("true",)), ("boolean",))),
            Atom(("_", ("=", ("Y",), ("true",)), ("boolean",))),
        ),
    )


def test_parse_formula_to_ast_prefix_unary():
    ast = parse_formula_to_ast("sum(Burglary): not =(Burglary,true)_boolean")
    assert ast == Aggregation(
        operation="sum",
        binders=(("Burglary",),),
        params=(),
        child=UnaryOp(
            "not", Atom(("_", ("=", ("Burglary",), ("true",)), ("boolean",)))
        ),
    )


def test_ast_binders_and_params_are_tuples():
    """Binders/params are tuples so nodes stay hashable/frozen."""
    ast = parse_formula_to_ast("sum(X, Y): p(X)_probability")
    assert isinstance(ast.binders, tuple)
    assert isinstance(ast.params, tuple)
    hash(ast)  # frozen dataclasses must be hashable


def test_circuit_node_is_a_formula_node_lump():
    """A ``CircuitNode`` folds as a leaf: ``_fold`` calls ``embed_circuit`` on it
    rather than recursing into AST children it does not have.

    The DeepProbLog engine emits already-compiled boolean lumps. The circuit
    factory embeds one verbatim (identity); the module factory *lowers* it to a
    runnable module. Either way it is treated as a leaf, never re-materialized.
    """
    factory = CircuitFactory()
    lump = factory.create_atom(("_", ("x",), ("boolean",)))
    assert isinstance(lump, CircuitNode)

    # The circuit factory embeds the lump verbatim (identity).
    assert fold(lump, factory) is lump

    # The module factory lowers the lump to a module (no longer a CircuitNode).
    lowered = fold(lump, DeepLogModuleFactory())
    assert isinstance(lowered, DeepLogModule)

    # A lump has no surface syntax, so the text interpreter rejects it.
    with pytest.raises(NotImplementedError):
        fold(lump, SymbolicFormulaFactory())


def test_structure_property_per_node_kind():
    """``.structure`` reports the algebraic structure carried by each node kind."""
    boolean = ("_", ("=", ("X",), ("true",)), ("boolean",))
    prob = ("_", ("p", ("X",)), ("probability",))

    assert Atom(boolean).structure == "boolean"
    assert Atom(prob).structure == "probability"
    assert Atom(("x",)).structure is None  # bare, unwrapped atom

    assert UnaryOp("not", Atom(boolean)).structure == "boolean"
    assert BinaryOp("or", Atom(boolean), Atom(boolean)).structure == "boolean"
    # disagreeing operands -> structure is unknown
    assert BinaryOp("or", Atom(boolean), Atom(prob)).structure is None

    assert Transformation("probability", Atom(boolean)).structure == "probability"
    # an expectation yields probability; any other aggregation carries its child
    assert Aggregation("expectation", (("X",),), (), Atom(boolean)).structure == (
        "probability"
    )
    assert Aggregation("sum", (("X",),), (), Atom(prob)).structure == "probability"

    # a CircuitNode reports its backing circuit's structure
    assert CircuitFactory().create_atom(boolean).structure == "boolean"


def test_parse_interns_shared_subformula():
    """``parse_formula_to_ast`` returns a canonical DAG: equal subformulas share.

    Two textually-identical operands collapse to the *same object* (``is``), yet
    the AST is still structurally equal to the expected tree — interning changes
    identity, never value.
    """
    ast = parse_formula_to_ast("(=(X,true)_boolean or =(X,true)_boolean)_probability")

    atom = Atom(("_", ("=", ("X",), ("true",)), ("boolean",)))
    assert ast == Transformation("probability", BinaryOp("or", atom, atom))

    # The two operands are interned to one object, not merely equal.
    assert isinstance(ast, Transformation)
    binary = ast.child
    assert isinstance(binary, BinaryOp)
    assert binary.lhs is binary.rhs


def test_circuit_node_lives_in_ast():
    """``CircuitNode`` is defined in the AST layer and re-exported unchanged."""
    from deeplog.formula.ast import CircuitNode as AstCircuitNode

    assert AstCircuitNode is CircuitNode

    lump = CircuitFactory().create_atom(("_", ("x",), ("boolean",)))
    assert isinstance(lump, AstCircuitNode)
    # A CircuitNode compresses a chunk; children are the reachable boundary
    # atoms of that chunk, so a leaf-backed chunk exposes itself.
    assert lump.children == (Atom(("_", ("x",), ("boolean",))),)


def test_circuit_node_children_are_reachable_boundary_atoms():
    """CircuitNode children are all atom leaves feeding the compressed chunk."""
    factory = CircuitFactory()
    b_sym = ("_", ("=", ("B",), ("true",)), ("boolean",))
    e_sym = ("_", ("=", ("E",), ("true",)), ("boolean",))

    negated_e = factory.create_unary_node("not", factory.create_atom(e_sym))
    conjunction = factory.create_binary_node(
        "and", factory.create_atom(b_sym), negated_e
    )

    assert conjunction.children == (Atom(b_sym), Atom(e_sym))


def test_circuit_node_children_decode_constants_as_atoms():
    """Named and numeric constants stay formula-level atoms in child views."""
    factory = CircuitFactory()
    x_sym = ("_", ("x",), ("boolean",))
    zero_sym = ("_", ("false",), ("boolean",))
    half_sym = ("_", ("0.5",), ("probability",))
    p_sym = ("_", ("p",), ("probability",))

    bool_node = factory.create_binary_node(
        "or", factory.create_atom(x_sym), factory.create_atom(zero_sym)
    )
    prob_node = factory.create_binary_node(
        "times", factory.create_atom(p_sym), factory.create_atom(half_sym)
    )

    assert bool_node.children == (Atom(x_sym), Atom(zero_sym))
    assert prob_node.children == (Atom(p_sym), Atom(half_sym))


def test_map_children_remaps_circuit_node_boundary_atoms():
    """CircuitNode maps boundary atoms by rebuilding the compressed circuit."""
    factory = CircuitFactory()
    b_sym = ("_", ("=", ("B",), ("true",)), ("boolean",))
    e_sym = ("_", ("=", ("E",), ("true",)), ("boolean",))
    c_sym = ("_", ("=", ("C",), ("true",)), ("boolean",))
    node = factory.create_binary_node(
        "and", factory.create_atom(b_sym), factory.create_atom(e_sym)
    )

    mapped = map_children(
        node,
        lambda child: Atom(c_sym) if child == Atom(b_sym) else child,
    )

    assert isinstance(mapped, CircuitNode)
    assert mapped.children == (Atom(c_sym), Atom(e_sym))
    assert node.children == (Atom(b_sym), Atom(e_sym))


def test_map_children_allows_non_atom_circuit_node_child_rewrite():
    """A boundary child may be rewritten to an arbitrary (non-atom) sub-formula.

    The symbolic boundary records the rewrite as a ``feeders`` override; the
    circuit graph is untouched, and the child view reports the new sub-formula.
    """
    x_sym = ("_", ("x",), ("boolean",))
    node = CircuitFactory().create_atom(x_sym)

    mapped = map_children(node, lambda child: UnaryOp("not", child))

    assert isinstance(mapped, CircuitNode)
    assert mapped.children == (UnaryOp("not", Atom(x_sym)),)
    assert mapped.feeders == ((x_sym, UnaryOp("not", Atom(x_sym))),)
    # The original lump is untouched — the rewrite is a fresh value.
    assert node.children == (Atom(x_sym),)


def test_map_children_remaps_circuit_node_constant_boundary():
    """A constant boundary child can be fed by another sub-formula too.

    Constants are boundary children like any leaf, so the symbolic boundary
    rewrites them without touching the circuit graph.
    """
    factory = CircuitFactory()
    x_sym = ("_", ("x",), ("boolean",))
    zero_sym = ("_", ("false",), ("boolean",))
    one_sym = ("_", ("true",), ("boolean",))
    node = factory.create_binary_node(
        "or", factory.create_atom(x_sym), factory.create_atom(zero_sym)
    )

    mapped = map_children(
        node,
        lambda child: Atom(one_sym) if child == Atom(zero_sym) else child,
    )

    assert isinstance(mapped, CircuitNode)
    assert mapped.children == (Atom(x_sym), Atom(one_sym))
    assert mapped.feeders == ((zero_sym, Atom(one_sym)),)


def test_fold_memo_shares_circuit_node():
    """``fold`` exploits the DAG: a textually-shared subformula folds once.

    The two ``(... and ...)`` groups are hash-consed to one object, so the
    per-call memo reuses its ``CircuitNode`` carrier — the circuit holds a single
    shared ``and`` node, not one per textual occurrence (the circuit dedups leaves
    but never operator nodes, so without the memo this would be two).
    """
    ast = parse_formula_to_ast(
        "(=(X,true)_boolean and =(Y,true)_boolean) or "
        "(=(X,true)_boolean and =(Y,true)_boolean)"
    )
    node = fold(ast, CircuitFactory())
    assert isinstance(node, CircuitNode)

    circuit = node.circuit
    and_nodes = sum(
        1
        for nid in circuit.iter_topological([node.node])
        if circuit._get_node(nid).node_type == "and"
    )
    assert and_nodes == 1


class _BoundaryRecorder(DeepLogFormulaFactory[str]):
    """A minimal lowering algebra that spells out what each eliminator saw."""

    lowers_circuit_children = True

    def create_atom(self, atom):
        return f"atom({atom[1][0]})"

    def create_binary_node(self, operator, lhs, rhs):
        return f"{operator}({lhs},{rhs})"

    def create_unary_node(self, operator, operand):
        return f"{operator}({operand})"

    def create_transformation(self, structure, child):
        return f"cast_{structure}({child})"

    def create_aggregation(self, operation, binders, params, child):
        return f"{operation}({child})"

    def embed_circuit(self, node, children=()):
        return f"lump[{','.join(children)}]"


def test_fold_keeps_boundary_atoms_distinct_across_lumps():
    """Each lump's boundary atom folds to *its own* carrier, not a neighbour's.

    ``CircuitNode.children`` materializes an ``Atom`` per boundary symbol, and
    the fold memo is keyed by object identity. Were those atoms rebuilt (and
    freed) on every access, CPython could hand a later atom a recycled address
    and the memo would answer with an earlier atom's carrier — silently wiring
    the wrong predicate onto a lump's boundary.
    """
    factory = CircuitFactory()
    lumps = [factory.create_atom(("_", (f"a{i}",), ("boolean",))) for i in range(3)]
    ast = BinaryOp("or", lumps[0], BinaryOp("or", lumps[1], lumps[2]))

    folded = fold(ast, _BoundaryRecorder())

    assert folded == "or(lump[atom(a0)],or(lump[atom(a1)],lump[atom(a2)]))"


def test_circuit_node_children_are_cached():
    """The boundary view is computed once per lump, so its atoms keep identity.

    It is read once per lump per AST pass; recomputing it would re-walk the
    reachable subgraph every time (and is what made the identity-keyed fold memo
    unsafe).
    """
    factory = CircuitFactory()
    lump = factory.create_atom(("_", ("a",), ("boolean",)))
    assert isinstance(lump, CircuitNode)

    assert lump.children is lump.children


def test_str_renders_the_formula_text_the_node_parsed_from():
    """``str`` is surface syntax: it round-trips back through the parser."""
    text = "(=(X,true)_boolean or =(Y,true)_boolean)_probability"
    ast = parse_formula_to_ast(text)

    assert str(ast) == text
    assert parse_formula_to_ast(str(ast)) == ast


def test_repr_renders_the_node_structure_on_one_line():
    """``repr`` names each node kind and the datum that distinguishes it."""
    ast = parse_formula_to_ast("sum(X): not p(X)_probability")

    assert repr(ast) == "Aggregation(sum(X), UnaryOp(not, Atom(p(X)_probability)))"


def test_tree_indents_one_line_per_node():
    """``tree`` is ``repr``'s structure, one node per line, children under parents."""
    ast = parse_formula_to_ast("(=(X,true)_boolean or =(Y,true)_boolean)_probability")

    assert ast.tree() == (
        "Transformation probability\n"
        "└─ BinaryOp or\n"
        "   ├─ Atom =(X,true)_boolean\n"
        "   └─ Atom =(Y,true)_boolean"
    )


def test_renderings_of_a_lump_show_its_boundary_but_not_its_interior():
    """A lump has no surface syntax, so ``str`` names it and the structure shows
    the formulas its boundary defers — never the compiled interior."""
    lump = fold(parse_formula_to_ast("not =(X,true)_boolean"), CircuitFactory())
    assert isinstance(lump, CircuitNode)
    handle = f"{lump.circuit.name}#{lump.node}"

    assert str(lump) == handle
    assert repr(lump) == f"CircuitNode({handle}, Atom(=(X,true)_boolean))"
    assert lump.tree() == f"CircuitNode {handle}\n└─ Atom =(X,true)_boolean"


def test_ast_factory_is_the_identity_interpreter():
    """Folding through ``AstFactory`` reproduces the node it started from."""
    ast = parse_formula_to_ast(
        "sum(X; q(X)_probability): (p(X)_boolean or not r(X)_boolean)_probability"
    )

    assert fold(ast, AstFactory()) == ast


def test_ast_factory_splices_a_lump_in_verbatim():
    """A lump is already an AST node, so the identity interpreter returns it."""
    lump = fold(parse_formula_to_ast("not =(X,true)_boolean"), CircuitFactory())

    assert fold(lump, AstFactory()) is lump

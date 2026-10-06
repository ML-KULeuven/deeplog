#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the materialized formula AST — the shape the parser produces.

The fold (catamorphism) is exercised end-to-end elsewhere: ``str`` folds a
formula through the text interpreter, so the round-trip suites in
``test_text_parser`` and ``test_text_printer`` already drive it, and
the compiler tests cover the module path. This module therefore pins
only what is unique to the IR: that ``parse_formula`` materializes the expected,
immutable dataclasses, and what the fold guarantees on top of them.
"""

import pickle

import pytest

from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Compiler
from deeplog import DeepLogModule
from deeplog import Transformation
from deeplog import UnaryOp
from deeplog import parse_formula
from deeplog import with_structure
from deeplog.formula.ast import CircuitNode
from deeplog.formula.ast import fold
from deeplog.formula.ast import map_children
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.formula.symbolic_factory import SymbolicFormulaFactory


def test_parse_formula_aggregation_with_params():
    ast = parse_formula("expect(X; q(X)_probability): p(X)_probability")
    assert ast == Aggregation(
        operation="expect",
        binders=(("X",),),
        params=(Atom(("_", ("q", ("X",)), ("probability",))),),
        child=Atom(("_", ("p", ("X",)), ("probability",))),
    )


def test_parse_formula_binary_and_transformation():
    ast = parse_formula("(=(X,true)_boolean or =(Y,true)_boolean)_probability")
    assert ast == Transformation(
        "probability",
        BinaryOp(
            "or",
            Atom(("_", ("=", ("X",), ("true",)), ("boolean",))),
            Atom(("_", ("=", ("Y",), ("true",)), ("boolean",))),
        ),
    )


def test_parse_formula_prefix_unary():
    ast = parse_formula("sum(Burglary): not =(Burglary,true)_boolean")
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
    ast = parse_formula("sum(X, Y): p(X)_probability")
    assert isinstance(ast.binders, tuple)
    assert isinstance(ast.params, tuple)
    hash(ast)  # frozen dataclasses must be hashable


@pytest.mark.parametrize("binder", [("x",), ("f", ("X",))])
def test_an_aggregation_binds_only_variables(binder):
    """A binder is a variable: ``=`` would read a constant one as a constant."""
    body = Atom(with_structure(("=", binder, ("true",)), "boolean"))

    with pytest.raises(ValueError, match="not a variable"):
        Aggregation("sum", (binder,), (), Transformation("real", body))


def test_a_parsed_aggregation_binds_only_variables():
    with pytest.raises(ValueError, match="sum binds x, which is not a variable"):
        parse_formula("sum(x): (=(x,true)_boolean)_real")


def test_hashing_a_formula_does_not_recurse_into_it():
    node = Atom(with_structure(("a",), "probability"))
    for _ in range(10_000):
        node = UnaryOp("negate", node)

    assert hash(node) == hash(UnaryOp("negate", node.operand))


def test_a_pickled_node_equals_and_hashes_as_the_node():
    a = Atom(with_structure(("a",), "boolean"))
    node = BinaryOp("or", a, UnaryOp("not", a))

    loaded = pickle.loads(pickle.dumps(node))  # noqa: S301

    assert loaded == node
    assert hash(loaded) == hash(node)
    assert {node: 1}[loaded] == 1


def test_circuit_node_is_a_formula_node_lump():
    """A ``CircuitNode`` folds as a leaf: ``fold`` calls ``embed_circuit`` on it
    rather than recursing into AST children it does not have.

    The DeepProbLog engine emits already-compiled boolean lumps. The circuit
    factory embeds one verbatim (identity), and the compiler lowers it to a
    runnable module. Either way it is never re-materialized.
    """
    factory = CircuitFactory()
    lump = factory.create_atom(("_", ("x",), ("boolean",)))
    assert isinstance(lump, CircuitNode)

    # The circuit factory embeds the lump verbatim (identity).
    assert fold(lump, factory) is lump

    # The compiler lowers the lump to a module (no longer a CircuitNode).
    assert isinstance(Compiler().compile(lump), DeepLogModule)

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
    """``parse_formula`` returns a canonical DAG: equal subformulas share.

    Two textually-identical operands collapse to the *same object* (``is``), yet
    the AST is still structurally equal to the expected tree — interning changes
    identity, never value.
    """
    ast = parse_formula("(=(X,true)_boolean or =(X,true)_boolean)_probability")

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
    ast = parse_formula(
        "(=(X,true)_boolean and =(Y,true)_boolean) or "
        "(=(X,true)_boolean and =(Y,true)_boolean)"
    )
    node = fold(ast, CircuitFactory())
    assert isinstance(node, CircuitNode)

    circuit = node.circuit
    and_nodes = sum(
        1
        for nid in circuit._iter_topological([node.node])
        if circuit._get_node(nid).node_type == "and"
    )
    assert and_nodes == 1


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
    ast = parse_formula(text)

    assert str(ast) == text
    assert parse_formula(str(ast)) == ast


def test_both_spellings_of_an_equality_are_one_atom():
    """``=`` is symmetric: ``=(a, X)`` is the atom ``=(X, a)``."""
    written = Atom(with_structure(("=", ("X",), ("a",)), "boolean"))
    swapped = Atom(with_structure(("=", ("a",), ("X",)), "boolean"))

    assert swapped == written
    assert hash(swapped) == hash(written)
    assert swapped.atom == written.atom


@pytest.mark.parametrize(
    ("text", "printed"),
    [
        ("=(a,X)_boolean", "=(X,a)_boolean"),
        ("=(X,a)_boolean", "=(X,a)_boolean"),
        ("=(Y,X)_boolean", "=(Y,X)_boolean"),
        ("=(b,a)_boolean", "=(b,a)_boolean"),
    ],
)
def test_an_equality_prints_its_variable_before_a_written_value(text, printed):
    """Only a written value compared with a variable is reordered."""
    assert str(parse_formula(text)) == printed


def test_repr_renders_the_node_structure_on_one_line():
    """``repr`` names each node kind and the datum that distinguishes it."""
    ast = parse_formula("sum(X): not p(X)_probability")

    assert repr(ast) == "Aggregation(sum(X), UnaryOp(not, Atom(p(X)_probability)))"


def test_tree_indents_one_line_per_node():
    """``tree`` is ``repr``'s structure, one node per line, children under parents."""
    ast = parse_formula("(=(X,true)_boolean or =(Y,true)_boolean)_probability")

    assert ast.tree() == (
        "Transformation probability\n"
        "└─ BinaryOp or\n"
        "   ├─ Atom =(X,true)_boolean\n"
        "   └─ Atom =(Y,true)_boolean"
    )


def test_renderings_of_a_lump_show_its_boundary_but_not_its_interior():
    """A lump has no surface syntax, so ``str`` names it and the structure shows
    the formulas its boundary defers — never the compiled interior."""
    lump = fold(parse_formula("not =(X,true)_boolean"), CircuitFactory())
    assert isinstance(lump, CircuitNode)
    handle = f"{lump.circuit.name}#{lump.node}"

    assert str(lump) == handle
    assert repr(lump) == f"CircuitNode({handle}, Atom(=(X,true)_boolean))"
    assert lump.tree() == f"CircuitNode {handle}\n└─ Atom =(X,true)_boolean"


def test_fold_takes_a_formula_deeper_than_the_recursion_limit():
    """A grounded proof can chain thousands of disjunctions; the fold walks them.

    The chain is rendered too, since ``str`` folds through the text interpreter.
    """
    import sys

    depth = sys.getrecursionlimit() * 2
    chain = Atom(("_", ("a0",), ("boolean",)))
    for i in range(1, depth):
        chain = BinaryOp("or", chain, Atom(("_", (f"a{i}",), ("boolean",))))

    assert isinstance(fold(chain, CircuitFactory()), CircuitNode)
    assert str(chain).count(" or ") == depth - 1


def test_a_shared_memo_never_answers_for_a_node_that_is_gone():
    """Folds sharing one memo keep the nodes they folded alive.

    The memo is keyed by ``id``; were a folded node freed, a later node could get
    its ``id`` and be answered with the earlier node's result.
    """
    factory, memo = CircuitFactory(), {}
    lumps = [
        fold(Atom(("_", (name,), ("boolean",))), factory, memo=memo)
        for name in ("a", "b", "c")
    ]

    assert len({lump.node for lump in lumps}) == 3

#  Copyright (c) 2024-2026. KU Leuven
"""Tests for folding a circuit through a formula algebra.

A circuit is the atom/unary/binary fragment of the formula language, so folding
one needs no algebra of its own: the same
:class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory` that
:func:`~deeplog.formula.ast.fold` drives over a tree drives this over a graph.
"""

import pytest

from deeplog.algebraic import BOOLEAN
from deeplog.algebraic import PROBABILITY
from deeplog.circuit.circuit import Circuit
from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory


def _circuit() -> tuple[Circuit, int, int]:
    """``(a and b) or (not a)`` — ``a`` shared by two parents."""
    circuit = Circuit(BOOLEAN)
    a = circuit.get_leaf_node(("a",))
    b = circuit.get_leaf_node(("b",))
    root = circuit.get_operator("or")(
        circuit.get_operator("and")(a, b), circuit.get_operator("not")(a)
    )
    return circuit, root, a


def test_a_circuit_folds_through_the_formula_algebra():
    """A circuit re-emits as its formula text through the *unchanged* AST algebra.

    Nothing about ``SymbolicFormulaFactory`` knows what a circuit is; it is the
    algebra the textual language is folded through. That it also prints a
    circuit is the claim that a circuit is a fragment of the formula language,
    demonstrated rather than asserted in a docstring.
    """
    circuit, root, _ = _circuit()

    built = circuit.fold([root], SymbolicFormulaFactory())

    assert built[root] == "(a_boolean and b_boolean) or (not a_boolean)"


def test_a_shared_node_is_built_once():
    """The fold memoises by node id, so a shared subgraph is not duplicated.

    Circuit ids are canonical, so unlike the AST fold — which relies on
    hash-consing to make structurally equal subformulas the *same object* — this
    needs nothing established first.
    """
    circuit, root, a = _circuit()
    built: list[object] = []

    class _Counting(SymbolicFormulaFactory):
        def create_atom(self, atom):
            built.append(atom)
            return super().create_atom(atom)

    result = circuit.fold([root], _Counting())

    # ``a`` feeds both the conjunction and the negation, but is built once.
    assert built.count(("_", ("a",), ("boolean",))) == 1
    assert result[a] == "a_boolean"


def test_a_numeric_constant_folds_as_an_atom():
    """Leaves and constants alike fold through the one ``create_atom`` eliminator.

    ``get_symbol_name`` reports a symbol for a leaf, a named constant and an
    arbitrary numeric constant alike, which is what lets a single eliminator
    cover all three. (An *identity* constant rarely survives to be folded: the
    circuit absorbs ``times(p, 1)`` to ``p`` as it is built.)
    """
    circuit = Circuit(PROBABILITY)
    root = circuit.get_operator("times")(
        circuit.get_leaf_node(("p",)), circuit.get_leaf_node(("0.6",))
    )

    assert circuit.fold([root], SymbolicFormulaFactory())[root] == (
        "p_probability times 0.6_probability"
    )


def test_a_circuit_only_algebra_says_what_a_circuit_cannot_hold():
    """An algebra that only meets circuit nodes refuses the AST-only eliminators.

    A circuit has no aggregation, no cross-structure transformation and no lump
    to embed, so an algebra written for one raises rather than inventing an
    answer — and names itself, since the caller reached it by mistake.
    """
    from deeplog.circuit.knowledge_compile.diagram import DiagramAlgebra

    algebra = DiagramAlgebra(BOOLEAN, {})

    with pytest.raises(NotImplementedError, match="DiagramAlgebra"):
        algebra.create_aggregation("sum", [("X",)], (), object())
    with pytest.raises(NotImplementedError, match="no.*transformation"):
        algebra.create_transformation("probability", object())


def test_every_eliminator_is_still_abstract():
    """Refusing three of them is the algebra's choice, not the ABC's.

    The AST-only eliminators stay ``@abstractmethod``, so a formula factory that
    forgets one fails at class definition rather than at call time — which is
    what a circuit-only algebra trades three explicit raises for.
    """

    class _Incomplete(DeepLogFormulaFactory[str]):
        def create_atom(self, atom):
            return "atom"

        def create_unary_node(self, operator, operand):
            return "unary"

        def create_binary_node(self, operator, lhs, rhs):
            return "binary"

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()  # pyright: ignore[reportAbstractUsage]


def test_the_same_algebra_compiles_an_ast_and_a_circuit():
    """One algebra, driven by either fold — a circuit's or the AST's.

    ``DiagramAlgebra`` is a ``DeepLogFormulaFactory``, so nothing distinguishes
    the two sources to it: ``deeplog.formula.ast.fold`` drives it over a tree and
    ``Circuit.fold`` over a graph, and the same formula compiles to the same
    diagram either way. What is *not* shared is the pre-pass — a manager must be
    sized before its first literal exists, so each source collects its own atoms.
    """
    from pysdd.sdd import SddManager

    from deeplog.circuit.knowledge_compile.diagram import DiagramAlgebra
    from deeplog.formula.ast import fold
    from deeplog.formula.text_parser_lark import parse_formula_to_ast
    from deeplog.symbol import with_structure

    text = "(a_boolean and b_boolean) or (not a_boolean)"
    manager = SddManager(var_count=2, auto_gc_and_minimize=False)
    literals = {
        with_structure(name, "boolean"): manager.literal(variable)
        for variable, name in enumerate((("a",), ("b",)), start=1)
    }

    from_ast = fold(parse_formula_to_ast(text), DiagramAlgebra(BOOLEAN, literals))

    circuit, root, _ = _circuit()
    from_circuit = circuit.fold([root], DiagramAlgebra(BOOLEAN, literals))[root]

    # The manager canonicalises, so equal formulas *are* the same diagram node.
    assert from_ast.id == from_circuit.id
    assert from_ast.model_count() == 3

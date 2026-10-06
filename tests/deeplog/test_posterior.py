#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the scalar posterior P(q | e) = E[q∧e] / E[e].

A posterior is not an aggregation — it is a ``BinaryOp("divide", ...)`` over two
``expectation`` aggregations. ``divide`` is an operator of the probability
semifield like any other, so it becomes a circuit node; no backend has a node for
a quotient, so :mod:`deeplog.circuit.split` cuts the graph there and applies the
division across the compiled columns (:class:`~deeplog.module.ColumnwiseModule`).
These tests exercise that lowering end-to-end through
``Compiler.compile``.
"""

import pytest
import torch

from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Compiler
from deeplog import Transformation
from deeplog import parse_formula_to_module
from deeplog import reshape
from deeplog.shape import SymTensor


TRUE = ("true",)

BURGLARY = ("Burglary",)
EARTHQUAKE = ("Earthquake",)

BURGLARY_ATOM = ("=", BURGLARY, TRUE)
EARTHQUAKE_ATOM = ("=", EARTHQUAKE, TRUE)

BURGLARY_BOOL_SYM = ("_", BURGLARY_ATOM, ("boolean",))
EARTHQUAKE_BOOL_SYM = ("_", EARTHQUAKE_ATOM, ("boolean",))

BURGLARY_PROB_SYM = ("_", BURGLARY_ATOM, ("probability",))
EARTHQUAKE_PROB_SYM = ("_", EARTHQUAKE_ATOM, ("probability",))


def _expectation(binders, child):
    """Build an ``expectation`` AST node."""
    return Aggregation("expectation", tuple(binders), (), child)


def _and(lhs, rhs):
    """A boolean conjunction of two leaf symbols."""
    return BinaryOp("and", Atom(lhs), Atom(rhs))


def _posterior(joint, evidence):
    """A posterior is the division of two expectations: E[joint] / E[evidence]."""
    return BinaryOp("divide", joint, evidence)


def test_posterior_disjunction_evidence():
    """P(B | B∨E): joint B∧(B∨E)=B, evidence B∨E. With P(B)=0.8, P(E)=0.3."""
    # The joint q∧e = B ∧ (B∨E) simplifies to B.
    joint = _expectation([BURGLARY, EARTHQUAKE], Atom(BURGLARY_BOOL_SYM))
    evidence = _expectation(
        [BURGLARY, EARTHQUAKE],
        BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM)),
    )
    module = reshape(
        Compiler().compile(_posterior(joint, evidence)),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # P(B)=0.8 ; P(B∨E)=1-0.2*0.7=0.86 ; posterior = 0.8/0.86
    result = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(result, torch.tensor([[0.8 / 0.86]]))


def test_posterior_distinct_symbol_sets():
    """P(B | E): numerator over {B,E}, denominator over {E} — exercises symbol-union.

    E[B∧E]=0.24, E[E]=0.3 ⇒ 0.8. The denominator mentions only E, so the two
    operands cover different symbols and must be aligned on the union.
    """
    joint = _expectation(
        [BURGLARY, EARTHQUAKE],
        BinaryOp("and", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM)),
    )
    evidence = _expectation([EARTHQUAKE], Atom(EARTHQUAKE_BOOL_SYM))
    module = reshape(
        Compiler().compile(_posterior(joint, evidence)),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    result = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(result, torch.tensor([[0.8]]))


def test_posterior_impossible_evidence_stays_finite():
    """P(·|e) with P(e)=0 returns a finite (clamped) value rather than NaN/inf."""
    joint = _expectation(
        [BURGLARY, EARTHQUAKE],
        BinaryOp("and", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM)),
    )
    evidence = _expectation([EARTHQUAKE], Atom(EARTHQUAKE_BOOL_SYM))
    module = reshape(
        Compiler().compile(_posterior(joint, evidence)),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    result = module(torch.tensor([[0.8, 0.0]]))
    assert torch.isfinite(result).all()


def _weighted_count():
    """``sum(B): (B)_probability times p(B, 0.8)``: a count no circuit holds.

    A ``sum`` binds its variable, so it stays symbolic, and so does an operator
    over it.
    """
    weight = ("_", ("p", BURGLARY, ("_", ("0.8",), ("probability",))), ("probability",))
    weighted = BinaryOp(
        "times",
        Transformation("probability", Atom(BURGLARY_BOOL_SYM)),
        Atom(weight),
    )
    return Aggregation("sum", (BURGLARY,), (), weighted)


def test_symbolic_operator_the_structure_defines_lowers_elementwise():
    """A ``times`` over two already-reduced counts multiplies their outputs.

    The operands are not lumps, so the construction fold could not absorb the
    ``times`` into a circuit; what reaches the lowering is a product of two
    numbers, and the probability algebra supplies the multiplication.
    """
    count = _weighted_count()
    product = Compiler().compile(BinaryOp("times", count, count))
    # The count is 0.8, so the product is 0.64.
    torch.testing.assert_close(product(), torch.tensor([[0.64]]))


def test_operator_the_structure_does_not_define_is_rejected():
    """An operator absent from the algebra names itself in the error."""
    count = _weighted_count()
    with pytest.raises(NotImplementedError, match="has no operator 'implies'"):
        Compiler().compile(BinaryOp("implies", count, count))


# --- End-to-end through the parser's `/` syntax ---


def _wmc(joint: bool) -> str:
    """A canonical probability WMC for B∧E (joint) or E (evidence)."""
    if joint:
        return (
            "(sum(Burglary, Earthquake): "
            "(=(Burglary,true)_boolean and =(Earthquake,true)_boolean)_probability "
            "times (=(Burglary,true)_probability times =(Earthquake,true)_probability))"
        )
    return (
        "(sum(Earthquake): "
        "(=(Earthquake,true)_boolean)_probability "
        "times (=(Earthquake,true)_probability))"
    )


def test_posterior_via_parser_divide_syntax():
    """`(WMC_q∧e) divide (WMC_e)` parses and evaluates to P(B|E)=0.8."""
    text = f"{_wmc(True)} divide {_wmc(False)}"
    module = reshape(
        parse_formula_to_module(text),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )
    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.3]])), torch.tensor([[0.8]])
    )


def test_parser_divide_that_is_no_posterior_is_a_plain_ratio():
    """A `divide` of two counts is the ratio of their values.

    The numerator's evidence is not a conjunct of it, so this is no posterior.
    The binders range over ``false, true`` and the weights do not depend on
    them: the disjunction holds in three of the four models, each weighing
    ``0.8 * 0.3``, and the evidence in one, weighing ``0.3``.
    """
    disjunctive = (
        "(sum(Burglary, Earthquake): "
        "(=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability "
        "times (=(Burglary,true)_probability times =(Earthquake,true)_probability))"
    )
    module = reshape(
        parse_formula_to_module(f"{disjunctive} divide {_wmc(False)}"),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )
    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.3]])), torch.tensor([[3 * 0.8 * 0.3 / 0.3]])
    )


def test_operator_above_a_division_composes():
    """``times(divide(a, b), c)`` — an *internal* division.

    The shape of a shielded policy: ``(π⁺ · P(safe|s,a)).sum()`` is a product
    sitting above a normalisation. The division breaks the circuit, so
    everything above it stays symbolic and is applied elementwise — one
    compiled circuit per division-free subtree, a tensor expression above.
    """
    compiler = Compiler()
    joint = _expectation(
        [BURGLARY, EARTHQUAKE], _and(BURGLARY_BOOL_SYM, EARTHQUAKE_BOOL_SYM)
    )
    evidence = _expectation([EARTHQUAKE], Atom(EARTHQUAKE_BOOL_SYM))
    other = _expectation([BURGLARY], Atom(BURGLARY_BOOL_SYM))

    module = reshape(
        compiler.compile(BinaryOp("times", _posterior(joint, evidence), other)),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    # P(B|E) = 0.24/0.3 = 0.8, times E[B] = 0.8 -> 0.64
    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.3]])), torch.tensor([[0.64]])
    )


def test_division_nested_under_a_division():
    """A divide whose numerator is itself a divide lowers as nested tensor ops."""
    compiler = Compiler()
    joint = _expectation(
        [BURGLARY, EARTHQUAKE], _and(BURGLARY_BOOL_SYM, EARTHQUAKE_BOOL_SYM)
    )
    evidence = _expectation([EARTHQUAKE], Atom(EARTHQUAKE_BOOL_SYM))
    other = _expectation([BURGLARY], Atom(BURGLARY_BOOL_SYM))

    module = reshape(
        compiler.compile(_posterior(_posterior(joint, evidence), other)),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.3]])), torch.tensor([[1.0]])
    )

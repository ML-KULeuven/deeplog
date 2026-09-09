"""Parity tests verifying SimpleGrounder and JanusGrounder produce identical formulas.

Both engines are driven through :class:`~deeplog.formula.ast_factory.AstFactory`,
so a proof is compared as the formula it is rather than as text.
"""

#  Copyright (c) 2024-2026. KU Leuven

import math
from functools import reduce

import pytest


pytest.importorskip("janus_swi")

from deeplog.formula import AstFactory
from deeplog.formula import BinaryOp
from deeplog.formula import FormulaNode
from deeplog.formula import map_children
from deeplog.grounding import Builtin
from deeplog.grounding import JanusGrounder
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import str_to_rules
from deeplog.symbol import is_variable
from deeplog.systems.deepproblog import Solver

from ...testing_formulas import operands


def _canonicalize(node: FormulaNode) -> FormulaNode:
    """Return ``node`` with every disjunction flattened and its operands sorted.

    Disjunction is commutative and associative, so two engines may legitimately
    reach the same formula having enumerated the clauses in a different order.
    Everything else about a proof is structure the two must agree on exactly.
    """
    node = map_children(node, _canonicalize)
    if isinstance(node, BinaryOp) and node.operator == "or":
        return reduce(
            lambda lhs, rhs: BinaryOp("or", lhs, rhs),
            sorted(operands(node, "or"), key=repr),
        )
    return node


def _assert_results_match(simple_result: dict, janus_result: dict) -> None:
    assert set(simple_result.keys()) == set(janus_result.keys()), (
        f"Key mismatch.\n"
        f"  SimpleGrounder keys: {set(simple_result.keys())}\n"
        f"  JanusGrounder keys:  {set(janus_result.keys())}"
    )

    for key in simple_result:
        s, j = simple_result[key], janus_result[key]
        if s == j:
            continue
        assert _canonicalize(s) == _canonicalize(j), (
            f"Formula mismatch for {key}.\n  SimpleGrounder: {s!r}\n  JanusGrounder:  {j!r}"
        )


def assert_engine_parity(code: str) -> None:
    """Assert both engines produce identical formula dicts for the given program."""
    program = tuple(str_to_rules(code))
    factory = AstFactory()

    simple_result = Solver(SimpleGrounder()).get_query_result(program, factory)
    janus_result = Solver(JanusGrounder()).get_query_result(program, factory)

    _assert_results_match(simple_result.formulas, janus_result.formulas)


def assert_conditional_parity(code: str) -> None:
    """Assert both engines produce identical conditional results (formulas + evidence)."""
    program = tuple(str_to_rules(code))
    factory = AstFactory()

    simple_result = Solver(SimpleGrounder()).get_query_result(program, factory)
    janus_result = Solver(JanusGrounder()).get_query_result(program, factory)

    _assert_results_match(simple_result.formulas, janus_result.formulas)
    assert (simple_result.evidence is None) == (janus_result.evidence is None)
    if simple_result.evidence is not None:
        assert _canonicalize(simple_result.evidence) == _canonicalize(
            janus_result.evidence
        )


def assert_engine_parity_with_builtins(
    code: str,
    builtins: dict[tuple[str, int], Builtin],
) -> None:
    """Assert parity when custom builtins are registered on both engines."""
    program = tuple(str_to_rules(code))
    factory = AstFactory()

    simple = Solver(SimpleGrounder())
    janus = Solver(JanusGrounder())
    for (functor, arity), fn in builtins.items():
        simple.add_builtin(functor, arity, fn)
        janus.add_builtin(functor, arity, fn)

    simple_result = simple.get_query_result(program, factory)
    janus_result = janus.get_query_result(program, factory)

    _assert_results_match(simple_result.formulas, janus_result.formulas)


# -- Basic facts and rules --


def test_single_labeled_fact():
    assert_engine_parity("""
        la::a.
        ?- a.
    """)


def test_conjunction():
    assert_engine_parity("""
        la::a.
        lb::b.
        ?- a, b.
    """)


def test_disjunction_via_clauses():
    assert_engine_parity("""
        a :- b.
        a :- c.
        lb::b.
        lc::c.
        ?- a.
    """)


def test_negation():
    assert_engine_parity("""
        la::a.
        ?- not(a).
    """)


def test_negation_of_conjunction():
    assert_engine_parity("""
        la::a.
        lb::b.
        ab :- a, b.
        ?- not(ab).
    """)


def test_rule_chaining():
    assert_engine_parity("""
        a :- b.
        lb::b.
        ?- a.
    """)


def test_deep_rule_chain():
    assert_engine_parity("""
        a :- b.
        b :- c.
        lc::c.
        ?- a.
    """)


def test_identical_conjunction():
    assert_engine_parity("""
        la::a.
        ?- a, a.
    """)


# -- Variables and substitution --


def test_variable_in_query():
    assert_engine_parity("""
        la0::a(0).
        la1::a(1).
        ?- a(X).
    """)


def test_substitution():
    assert_engine_parity("""
        fact(t(1,2,X), t(2,1,X)).
        ?- fact(t(1,2,3), Z).
    """)


# -- Recursion --


def test_recursive_reasoning():
    assert_engine_parity("""
        edge(0,1).
        edge(1,2).
        edge(1,3).

        connected(X,Y) :- edge(X,Y).
        connected(X,Y) :- edge(X,Z), connected(Z,Y).
        ?- connected(X,Y).
    """)


def test_recursive_with_labels():
    assert_engine_parity("""
        l01::edge(0,1).
        l12::edge(1,2).

        connected(X,Y) :- edge(X,Y).
        connected(X,Y) :- edge(X,Z), connected(Z,Y).
        ?- connected(X,Y).
    """)


# -- Builtins --


def test_labeled_rule_with_builtin():
    assert_engine_parity("""
        classifier(X) :: output(X) :- between(0,9,X).
        ?- output(Y).
    """)


def test_addition_program():
    assert_engine_parity("""
        addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).
        classifier(I,N) :: digit(I,N).
        ?- addition(i1,i2,S).
    """)


def test_custom_builtin():
    def square(lhs, rhs):
        if is_variable(lhs):
            if not is_variable(rhs):
                yield {lhs: (str(math.sqrt(int(rhs[0]))),)}
        else:
            if is_variable(rhs):
                yield {rhs: (str(int(lhs[0]) ** 2),)}
            else:
                if int(rhs[0]) == int(lhs[0]) ** 2:
                    yield {}

    assert_engine_parity_with_builtins(
        "?- square(2,X).",
        builtins={("square", 2): square},
    )


# -- Multiple queries --


def test_multiple_queries():
    assert_engine_parity("""
        la::a.
        lb::b.
        ?- a.
        ?- b.
    """)


# -- Lists --


def test_list_predicates():
    assert_engine_parity("""
        cons([H|T], H, T).
        head(L, H) :- cons(L, H, _).
        tail(L, T) :- cons(L, _, T).
        ?- head([a,b,c], H).
        ?- tail([a,b,c], T).
    """)


def test_multi_digit_addition():
    assert_engine_parity("""
        classifier(I,N) :: classify(I,N).
        digit(I,N) :- between(0,9,N), classify(I,N).

        add(I1, I2, C, N) :- between(0, 1, C), digit(I1, N1), digit(I2, N2), is(N, mod(+(N1,+(N2,C)), 10)).
        carry(I1, I2, C, Cout) :- between(0, 1, C), digit(I1, N1), digit(I2, N2), is(Cout, div(+(N1,+(N2,C)), 10)).

        carry(nil, nil, 0).
        carry(list(H1, T1), list(H2, T2), Cout) :- carry(T1, T2, Cin), carry(H1, H2, Cin, Cout).

        addition(L1, L2, -1, N) :- carry(L1, L2, N).
        addition(list(H1, T1), list(H2, T2), 0, N) :- carry(T1, T2, C), add(H1, H2, C, N).
        addition(list(A, T1), list(B, T2), Idx, N) :- between(1, 2, Idx), is(Idx2, -(Idx, 1)), addition(T1, T2, Idx2, N).

        ?- addition(list(a,list(b,nil)),list(c,list(d,nil)),-1,X).
        ?- addition(list(a,list(b,nil)),list(c,list(d,nil)),0,X).
        ?- addition(list(a,list(b,nil)),list(c,list(d,nil)),1,X).
    """)


def test_double_negation():
    assert_engine_parity("""
        la::a.
        ?- not(not(a)).
    """)


def test_negation_of_true_fact():
    assert_engine_parity("""
        a.
        ?- not(a).
    """)


# -- Conditional queries (evidence via integrity constraints) --


def test_conditional_positive_evidence_parity():
    assert_conditional_parity("""
        0.6::burglary.
        0.3::earthquake.
        alarm :- burglary.
        alarm :- earthquake.
        :- not(alarm).
        ?- burglary.
    """)


def test_conditional_negative_evidence_parity():
    assert_conditional_parity("""
        0.6::burglary.
        0.3::earthquake.
        alarm :- burglary.
        alarm :- earthquake.
        :- earthquake.
        ?- burglary.
        ?- alarm.
    """)

#  Copyright (c) 2024-2026. KU Leuven
"""Parity tests: SimpleGrounder and JanusGrounder build the same proofs.

Both grounders build through :class:`~deeplog.formula.ast_factory.AstFactory`, so a
proof is compared as the formula it is rather than as text. Every query and every
constraint body of a program is grounded through one
:class:`~deeplog.grounding.ProofBuilder` per grounder, the way a caller conditioning
on evidence grounds it, and both builders must end up with the same leaves.
"""

import math
from collections.abc import Iterable
from functools import reduce

import pytest


pytest.importorskip("janus_swi")

from deeplog import Symbol
from deeplog import is_variable
from deeplog.formula import AstFactory
from deeplog.formula import BinaryOp
from deeplog.formula import FormulaNode
from deeplog.formula import map_children
from deeplog.grounding import JanusGrounder
from deeplog.grounding import PrologGrounder
from deeplog.grounding import ProofBuilder
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import str_to_rules
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_query

from ...testing_formulas import operands


def _canonicalize(node: FormulaNode) -> FormulaNode:
    """Return ``node`` with every disjunction flattened and its operands sorted.

    Disjunction is commutative and associative, so two grounders may legitimately
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


def _goals(program: Iterable[Symbol]) -> Iterable[Symbol]:
    """The goal of every query and the body of every constraint, in order."""
    for clause in program:
        if is_query(clause):
            yield clause[2]
        elif is_constraint(clause):
            yield get_constraint_body(clause)


def _ground(grounder: PrologGrounder, code: str, open_predicates):
    """Every goal's proofs, and the leaves the shared builder saw."""
    program = tuple(str_to_rules(code))
    builder = ProofBuilder(AstFactory())
    proofs = [
        {
            answer: _canonicalize(proof)
            for answer, proof in grounder.ground(
                program, goal, builder, open_predicates
            ).items()
        }
        for goal in _goals(program)
    ]
    return proofs, builder.leaves


def _assert_parity(code: str, open_predicates, simple=None, janus=None) -> None:
    simple_proofs, simple_leaves = _ground(
        simple or SimpleGrounder(), code, open_predicates
    )
    janus_proofs, janus_leaves = _ground(
        janus or JanusGrounder(), code, open_predicates
    )
    assert all(simple_proofs), "every goal of these programs has an answer"
    assert simple_proofs == janus_proofs
    assert simple_leaves == janus_leaves


PROGRAMS = {
    "fact": ("a.\n?- a.", {("a", 0)}),
    "conjunction": ("a.\nb.\n?- a, b.", {("a", 0), ("b", 0)}),
    "identical_conjunction": ("a.\n?- a, a.", {("a", 0)}),
    "disjunction_over_clauses": (
        "a :- b.\na :- c.\nb.\nc.\n?- a.",
        {("b", 0), ("c", 0)},
    ),
    "disjunction_in_a_body": ("q :- a ; b.\na.\nb.\n?- q.", {("a", 0), ("b", 0)}),
    "negation": ("a.\n?- not(a).", {("a", 0)}),
    "negation_of_a_conjunction": (
        "a.\nb.\nab :- a, b.\n?- not(ab).",
        {("a", 0), ("b", 0)},
    ),
    "double_negation": ("a.\n?- not(not(a)).", {("a", 0)}),
    "negation_of_a_closed_fact": ("a.\n?- not(a).", set()),
    "rule_chain": ("a :- b.\nb :- c.\nc.\n?- a.", {("c", 0)}),
    "variable_in_query": ("a(0).\na(1).\n?- a(X).", {("a", 1)}),
    "substitution": ("fact(t(1,2,X), t(2,1,X)).\n?- fact(t(1,2,3), Z).", set()),
    "recursion": (
        "edge(0,1).\nedge(1,2).\nedge(1,3).\n"
        "connected(X,Y) :- edge(X,Y).\n"
        "connected(X,Y) :- edge(X,Z), connected(Z,Y).\n"
        "?- connected(X,Y).",
        set(),
    ),
    "recursion_over_open_facts": (
        "edge(0,1).\nedge(1,2).\n"
        "connected(X,Y) :- edge(X,Y).\n"
        "connected(X,Y) :- edge(X,Z), connected(Z,Y).\n"
        "?- connected(X,Y).",
        {("edge", 2)},
    ),
    "open_rule_over_a_builtin": (
        "output(X) :- between(0,9,X).\n?- output(Y).",
        {("output", 1)},
    ),
    "open_rule_with_a_quoted_name": (
        "'an output'(X) :- between(0,1,X).\n?- 'an output'(Y).",
        {("'an output'", 1)},
    ),
    "addition": (
        "addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), "
        "digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).\n"
        "digit(I,N).\n"
        "?- addition(i1,i2,S).",
        {("digit", 2)},
    ),
    "multi_digit_addition": (
        """
        classify(I,N).
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
        """,
        {("classify", 2)},
    ),
    "lists": (
        "cons([H|T], H, T).\n"
        "head(L, H) :- cons(L, H, _).\n"
        "tail(L, T) :- cons(L, _, T).\n"
        "?- head([a,b,c], H).\n"
        "?- tail([a,b,c], T).",
        set(),
    ),
    "empty_list": ("empty([]).\n?- empty(X).", set()),
    "anonymous_variables": (
        "p(_, _).\nr(a, b).\nr(c, a).\nq(X) :- r(X, _), r(_, X).\n"
        "?- p(a, b).\n?- q(a).\n?- r(_, _).",
        set(),
    ),
    "anonymous_variables_in_open_facts": ("p(a, _).\n?- p(a, b).", {("p", 2)}),
    "constraints": (
        "burglary.\nearthquake.\n"
        "alarm :- burglary.\nalarm :- earthquake.\n"
        ":- not(alarm).\n:- earthquake.\n"
        "?- burglary.\n?- alarm.",
        {("burglary", 0), ("earthquake", 0)},
    ),
}


@pytest.mark.parametrize(("code", "open_predicates"), PROGRAMS.values(), ids=PROGRAMS)
def test_grounders_agree(code, open_predicates):
    _assert_parity(code, open_predicates)


def test_grounders_agree_through_an_added_builtin():
    def square(lhs, rhs):
        if is_variable(lhs):
            if not is_variable(rhs):
                yield {lhs: (str(math.isqrt(int(rhs[0]))),)}
        elif is_variable(rhs):
            yield {rhs: (str(int(lhs[0]) ** 2),)}
        elif int(rhs[0]) == int(lhs[0]) ** 2:
            yield {}

    simple, janus = SimpleGrounder(), JanusGrounder()
    for grounder in (simple, janus):
        grounder.add_builtin("square", 2, square)
    _assert_parity(
        "four(X) :- square(2,X).\n?- square(3,X).\n?- four(X).\n?- square(2,4).",
        set(),
        simple=simple,
        janus=janus,
    )

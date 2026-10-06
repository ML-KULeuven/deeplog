#  Copyright (c) 2024-2026. KU Leuven
"""Plain-Prolog grounder tests — zero ProbLog / probability involvement.

These pin down the grounder contract in isolation: it takes a plain program, a
goal and a set of *open* predicates, and returns a boolean proof formula per
ground answer. Open-predicate facts become leaf atoms; every other fact collapses
to ``true``. No ``::``, no labels, no annotated disjunctions.
"""

import math

import pytest

from deeplog import Atom
from deeplog import BinaryOp
from deeplog.algebraic import BOOLEAN
from deeplog.grounding.prolog import JanusGrounder
from deeplog.grounding.prolog import SimpleGrounder
from deeplog.grounding.prolog import UnknownPredicateException
from deeplog.grounding.prolog import str_to_rules
from deeplog.symbol import is_variable
from deeplog.symbol import parse_symbol

from ...testing_formulas import leaf
from ...testing_formulas import operands


TRUE = Atom(("_", BOOLEAN.one, ("boolean",)))


def _grounders():
    grounders = [SimpleGrounder()]
    if JanusGrounder.is_available():
        grounders.append(JanusGrounder())
    return grounders


@pytest.mark.parametrize("grounder", _grounders())
def test_open_fact_becomesleaf(grounder):
    """An open predicate's fact is emitted as a boolean leaf atom."""
    program = tuple(str_to_rules("a.\nq :- a.\n?- q."))
    result = grounder.ground(program, parse_symbol("q"), open_predicates={("a", 0)})
    assert result == {("q",): leaf("a")}


@pytest.mark.parametrize("grounder", _grounders())
def test_closed_fact_collapses_to_true(grounder):
    """A closed (non-open) fact contributes ``true`` and folds away."""
    program = tuple(str_to_rules("a.\nb.\nq :- a, b.\n?- q."))
    # `a` is open (a leaf), `b` is closed (true) → q's proof is just the a-leaf.
    result = grounder.ground(program, parse_symbol("q"), open_predicates={("a", 0)})
    assert result == {("q",): leaf("a")}


@pytest.mark.parametrize("grounder", _grounders())
def test_explicit_true_goal_is_the_identity_of_conjunction(grounder):
    """``true`` in a body contributes nothing; it must not zero the conjunction."""
    program = tuple(str_to_rules("a.\nq :- true, a.\n?- q."))
    result = grounder.ground(program, parse_symbol("q"), open_predicates={("a", 0)})
    assert result == {("q",): leaf("a")}


@pytest.mark.parametrize("grounder", _grounders())
def test_disjunction_over_clauses(grounder):
    """Two clauses for the same head OR-fold into one proof formula."""
    program = tuple(str_to_rules("q :- a.\nq :- b.\na.\nb.\n?- q."))
    result = grounder.ground(
        program,
        parse_symbol("q"),
        open_predicates={("a", 0), ("b", 0)},
    )
    assert set(operands(result[("q",)], "or")) == {leaf("a"), leaf("b")}


@pytest.mark.parametrize("grounder", _grounders())
def test_variable_query_enumerates_answers(grounder):
    """A variable goal yields one entry per ground answer."""
    program = tuple(str_to_rules("p(1).\np(2).\n?- p(X)."))
    result = grounder.ground(
        program,
        parse_symbol("p(X)"),
        open_predicates={("p", 1)},
    )
    assert set(result.keys()) == {("p", ("1",)), ("p", ("2",))}


@pytest.mark.parametrize("grounder", _grounders())
def test_default_open_set_is_empty(grounder):
    """With no open predicates every fact is closed → the proof is ``true``."""
    program = tuple(str_to_rules("a.\nq :- a.\n?- q."))
    result = grounder.ground(program, parse_symbol("q"))
    # No leaves at all: q reduces to the boolean-true constant atom, which bakes
    # to 1 downstream.
    assert result == {("q",): TRUE}


@pytest.mark.parametrize("grounder", _grounders())
def test_unknown_predicate_raises(grounder):
    program = tuple(str_to_rules("?- missing."))
    with pytest.raises(UnknownPredicateException):
        grounder.ground(program, parse_symbol("missing"))


@pytest.mark.parametrize("grounder", _grounders())
def test_builtin_binds_and_never_leaks_aleaf(grounder):
    """`between/3` gates resolution and never becomes a leaf itself."""
    program = tuple(
        str_to_rules("r(X) :- between(0, 2, X), p(X).\np(0).\np(1).\np(2).")
    )
    result = grounder.ground(
        program,
        parse_symbol("r(X)"),
        open_predicates={("p", 1)},
    )
    assert set(result.keys()) == {("r", ("0",)), ("r", ("1",)), ("r", ("2",))}
    for formula in result.values():
        # Only p-leaves appear; between/3 left no atom behind.
        assert "between" not in str(formula)


@pytest.mark.parametrize("grounder", _grounders())
def test_open_predicate_defined_by_rule(grounder):
    """A rule for an open predicate declares its domain: instances are leaves."""
    program = tuple(str_to_rules("a(N) :- between(0,1,N).\nq(N) :- a(N)."))
    result = grounder.ground(
        program,
        parse_symbol("q(X)"),
        open_predicates={("a", 1)},
    )
    assert result == {
        ("q", ("0",)): leaf("a(0)"),
        ("q", ("1",)): leaf("a(1)"),
    }


@pytest.mark.parametrize("grounder", _grounders())
def test_open_rule_body_conjoins_withleaf(grounder):
    """An open predicate derived through open subgoals keeps both leaves."""
    program = tuple(str_to_rules("a(0).\nb(N) :- a(N).\nq(N) :- b(N)."))
    result = grounder.ground(
        program,
        parse_symbol("q(X)"),
        open_predicates={("a", 1), ("b", 1)},
    )
    assert result == {("q", ("0",)): BinaryOp("and", leaf("a(0)"), leaf("b(0)"))}


@pytest.mark.parametrize("grounder", _grounders())
def test_open_rule_nonground_head_raises(grounder):
    """A rule body that leaves the open head non-ground is an error."""
    program = tuple(str_to_rules("a(N) :- between(0,1,K), is(M, +(K,1)).\nq :- a(N)."))
    with pytest.raises(ValueError, match="not ground"):
        grounder.ground(
            program,
            parse_symbol("q"),
            open_predicates={("a", 1)},
        )


@pytest.mark.parametrize("grounder", _grounders())
def test_recursive_rules_reach_every_answer(grounder):
    program = tuple(
        str_to_rules(
            """
            edge(0,1).
            edge(1,2).
            edge(1,3).
            connected(X,Y) :- edge(X,Y).
            connected(X,Y) :- edge(X,Z), connected(Z,Y).
            """
        )
    )
    result = grounder.ground(program, parse_symbol("connected(X,Y)"))
    connected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    assert set(result) == {("connected", (str(x),), (str(y),)) for x, y in connected}


@pytest.mark.parametrize("grounder", _grounders())
def test_answer_substitutes_through_nested_terms(grounder):
    program = tuple(str_to_rules("fact(t(1,2,X), t(2,1,X))."))
    result = grounder.ground(program, parse_symbol("fact(t(1,2,3), Z)"))
    assert set(result) == {
        ("fact", ("t", ("1",), ("2",), ("3",)), ("t", ("2",), ("1",), ("3",)))
    }


@pytest.mark.parametrize("grounder", _grounders())
def test_added_builtin_binds_its_arguments(grounder):
    def square(lhs, rhs):
        if is_variable(lhs):
            if not is_variable(rhs):
                yield {lhs: (str(math.isqrt(int(rhs[0]))),)}
        elif is_variable(rhs):
            yield {rhs: (str(int(lhs[0]) ** 2),)}
        elif int(rhs[0]) == int(lhs[0]) ** 2:
            yield {}

    grounder.add_builtin("square", 2, square)
    result = grounder.ground((), parse_symbol("square(2,X)"))
    assert result == {parse_symbol("square(2,4)"): TRUE}


@pytest.mark.parametrize(
    "grounder_type",
    [SimpleGrounder, JanusGrounder]
    if JanusGrounder.is_available()
    else [SimpleGrounder],
)
def test_a_builtin_belongs_to_the_grounder_that_added_it(grounder_type):
    """Grounders proving one program each use only the builtins added to them."""

    def square(lhs, rhs):
        yield {rhs: (str(int(lhs[0]) ** 2),)}

    program = tuple(str_to_rules("q(Y) :- square(3, Y)."))
    goal = parse_symbol("q(Y)")
    without = grounder_type()
    with_square = grounder_type()
    with_square.add_builtin("square", 2, square)

    with pytest.raises(UnknownPredicateException):
        without.ground(program, goal)
    assert set(with_square.ground(program, goal)) == {parse_symbol("q(9)")}
    with pytest.raises(UnknownPredicateException):
        without.ground(program, goal)


@pytest.mark.parametrize("grounder", _grounders())
def test_list_terms_unify_through_rules(grounder):
    program = tuple(
        str_to_rules(
            """
            cons([H|T], H, T).
            head(L, H) :- cons(L, H, _).
            tail(L, T) :- cons(L, _, T).
            """
        )
    )
    abc = ("cons", ("a",), ("cons", ("b",), ("cons", ("c",), ("nil",))))
    bc = abc[2]

    heads = grounder.ground(program, parse_symbol("head([a,b,c], H)"))
    tails = grounder.ground(program, parse_symbol("tail([a,b,c], T)"))

    assert set(heads) == {("head", abc, ("a",))}
    assert set(tails) == {("tail", abc, bc)}


def test_a_goal_that_calls_itself_says_so():
    """Left recursion: proving the goal needs the goal, so resolution cannot."""
    program = tuple(
        str_to_rules(
            """
            edge(a,b).
            edge(X,Y) :- edge(X,Z), edge(Z,Y).
            """
        )
    )
    with pytest.raises(RecursionError, match="which is the same call"):
        SimpleGrounder().ground(program, parse_symbol("edge(a,c)"))


def test_a_rule_that_calls_itself_says_so():
    program = tuple(str_to_rules("p :- p."))
    with pytest.raises(RecursionError, match="Proving p calls p"):
        SimpleGrounder().ground(program, parse_symbol("p"))


def test_a_call_that_grows_its_term_is_proved():
    """Recursion that is not a variant of its caller terminates and is left alone."""
    program = tuple(str_to_rules("nat(0).\nnat(s(X)) :- nat(X)."))
    result = SimpleGrounder().ground(program, parse_symbol("nat(s(s(0)))"))
    assert set(result) == {("nat", ("s", ("s", ("0",))))}


def test_a_call_under_negation_is_checked_too():
    program = tuple(str_to_rules("q :- not(p).\np :- p."))
    with pytest.raises(RecursionError, match="Proving p calls p"):
        SimpleGrounder().ground(program, parse_symbol("q"))


@pytest.mark.parametrize("grounder", _grounders())
def test_an_answer_two_clauses_prove_is_proved_by_either(grounder):
    """No proof of an answer is dropped: they are disjoined."""
    program = tuple(str_to_rules("b.\nc.\na :- b.\na :- c.\n?- a."))
    result = grounder.ground(
        program, parse_symbol("a"), open_predicates={("b", 0), ("c", 0)}
    )
    assert result == {("a",): BinaryOp(BOOLEAN.sum, leaf("b"), leaf("c"))}

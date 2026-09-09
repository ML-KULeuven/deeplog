#  Copyright (c) 2024-2026. KU Leuven
"""Plain-Prolog grounder tests — zero ProbLog / probability involvement.

These pin down the grounder contract in isolation: it takes a plain program, a
goal, a :class:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory`,
and a set of *open* predicates, and returns a boolean proof structure per ground
answer. Open-predicate facts become leaf atoms; every other fact collapses to
``true``. No ``::``, no labels, no annotated disjunctions.

The factory is :class:`~deeplog.formula.ast_factory.AstFactory`, so each proof
is asserted as the formula the grounder built rather than as its rendering.
"""

import pytest

from deeplog.algebraic import BOOLEAN
from deeplog.formula import AstFactory
from deeplog.formula import Atom
from deeplog.formula import BinaryOp
from deeplog.formula import SymbolicFormulaFactory
from deeplog.grounding import JanusGrounder
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import UnknownPredicateException
from deeplog.grounding.prolog import str_to_rules
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
    result = grounder.ground(
        program, parse_symbol("q"), AstFactory(), open_predicates={("a", 0)}
    )
    assert result == {("q",): leaf("a")}


@pytest.mark.parametrize("grounder", _grounders())
def test_closed_fact_collapses_to_true(grounder):
    """A closed (non-open) fact contributes ``true`` and folds away."""
    program = tuple(str_to_rules("a.\nb.\nq :- a, b.\n?- q."))
    # `a` is open (a leaf), `b` is closed (true) → q's proof is just the a-leaf.
    result = grounder.ground(
        program, parse_symbol("q"), AstFactory(), open_predicates={("a", 0)}
    )
    assert result == {("q",): leaf("a")}


@pytest.mark.parametrize("grounder", _grounders())
def test_explicit_true_goal_is_the_identity_of_conjunction(grounder):
    """``true`` in a body contributes nothing; it must not zero the conjunction."""
    program = tuple(str_to_rules("a.\nq :- true, a.\n?- q."))
    result = grounder.ground(
        program, parse_symbol("q"), AstFactory(), open_predicates={("a", 0)}
    )
    assert result == {("q",): leaf("a")}


@pytest.mark.parametrize("grounder", _grounders())
def test_disjunction_over_clauses(grounder):
    """Two clauses for the same head OR-fold into one proof formula."""
    program = tuple(str_to_rules("q :- a.\nq :- b.\na.\nb.\n?- q."))
    result = grounder.ground(
        program,
        parse_symbol("q"),
        AstFactory(),
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
        AstFactory(),
        open_predicates={("p", 1)},
    )
    assert set(result.keys()) == {("p", ("1",)), ("p", ("2",))}


@pytest.mark.parametrize("grounder", _grounders())
def test_default_open_set_is_empty(grounder):
    """With no open predicates every fact is closed → the proof is ``true``."""
    program = tuple(str_to_rules("a.\nq :- a.\n?- q."))
    result = grounder.ground(program, parse_symbol("q"), AstFactory())
    # No leaves at all: q reduces to the boolean-true constant atom, which bakes
    # to 1 downstream.
    assert result == {("q",): TRUE}


@pytest.mark.parametrize("grounder", _grounders())
def test_unknown_predicate_raises(grounder):
    program = tuple(str_to_rules("?- missing."))
    with pytest.raises(UnknownPredicateException):
        grounder.ground(program, parse_symbol("missing"), AstFactory())


@pytest.mark.parametrize("grounder", _grounders())
def test_builtin_binds_and_never_leaks_aleaf(grounder):
    """`between/3` gates resolution and never becomes a leaf itself."""
    program = tuple(
        str_to_rules("r(X) :- between(0, 2, X), p(X).\np(0).\np(1).\np(2).")
    )
    result = grounder.ground(
        program,
        parse_symbol("r(X)"),
        AstFactory(),
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
        AstFactory(),
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
        AstFactory(),
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
            AstFactory(),
            open_predicates={("a", 1)},
        )


@pytest.mark.skipif(not JanusGrounder.is_available(), reason="janus_swi not installed")
def test_query_ignores_a_builder_an_earlier_query_left_behind():
    """A query that ended early must not steer the next query's aggregation.

    The Janus engine hands its tabled lattice aggregation a factory through a
    global fact, since the aggregation's arity has no room for one, and a query
    that ends early -- here, one that raises -- leaves that fact asserted. The
    next query must still aggregate through its own builder. Only two runs with
    *different* factories can tell the difference, and the fact is cleared first
    so it is this leak being read, not one queued up by an earlier test.
    """
    import janus_swi as janus

    janus.query_once("retractall(factory(_))")
    grounder = JanusGrounder()
    with pytest.raises(UnknownPredicateException):
        grounder.ground(
            tuple(str_to_rules("?- missing.")), parse_symbol("missing"), AstFactory()
        )

    program = tuple(str_to_rules("s :- m.\ns :- n.\nm.\nn.\n?- s."))
    result = grounder.ground(
        program,
        parse_symbol("s"),
        SymbolicFormulaFactory(),
        open_predicates={("m", 0), ("n", 0)},
    )

    assert set(result[("s",)].split(" or ")) == {"m_boolean", "n_boolean"}

#  Copyright (c) 2024-2026. KU Leuven
"""A prover written against DeepLog's Prolog library runs through a JanusProver."""

import re
from pathlib import Path

import pytest

from deeplog.grounding.prolog import JANUS_AVAILABLE
from deeplog.grounding.prolog import JanusProver
from deeplog.grounding.prolog import UnknownPredicateException
from deeplog.symbol import parse_symbol


pytestmark = pytest.mark.skipif(not JANUS_AVAILABLE, reason="janus_swi not installed")

HERE = Path(__file__).parent
DOCS = HERE.parents[4] / "site" / "source" / "docs" / "janus_prover.rst"

#: The library's public contract; changing it is a SemVer decision.
EXPORTS = {
    "to_symbol/2",
    "from_symbol/2",
    "is_builtin/2",
    "call_builtin/2",
    "unknown_predicate/1",
    "non_ground_open_predicate/1",
}

FACTS = [":- dynamic fact/1.", "fact(p(a)).", "fact(p(b)).", "fact(p([a,b]))."]


def _answers(prover: JanusProver, clauses: list[str], goal: str) -> set:
    rows = prover.query(
        "prove(Program, Prover, Goal, Answer)",
        {
            "Program": prover.consult(clauses),
            "Prover": prover,
            "Goal": parse_symbol(goal),
        },
    )
    return {row["Answer"] for row in rows}


def test_the_same_clauses_load_into_one_module():
    prover = JanusProver(HERE / "toy_engine.pl")
    other = JanusProver(HERE / "other_engine.pl")

    assert prover.consult(FACTS) == prover.consult(list(FACTS))
    assert other.consult(FACTS) == prover.consult(FACTS)
    assert prover.consult(FACTS[:2]) != prover.consult(FACTS)


def test_clauses_load_without_singleton_warnings(capfd):
    prover = JanusProver(HERE / "toy_engine.pl")

    prover.consult([":- dynamic fact/1.", "fact(singleton(X))."])

    assert "Singleton" not in capfd.readouterr().err


def test_a_query_yields_every_answer():
    prover = JanusProver(HERE / "toy_engine.pl")

    assert _answers(prover, FACTS, "p(X)") == {
        parse_symbol("p(a)"),
        parse_symbol("p(b)"),
        parse_symbol("p([a,b])"),
    }


def test_an_unknown_predicate_raises():
    prover = JanusProver(HERE / "toy_engine.pl")

    with pytest.raises(UnknownPredicateException, match=r"\('q', 1\)"):
        _answers(prover, FACTS, "q(X)")


def test_a_non_ground_open_predicate_raises():
    prover = JanusProver(HERE / "toy_engine.pl")

    with pytest.raises(ValueError, match=r"\('leaf', 1\) is not ground"):
        _answers(prover, FACTS, "leaf(X)")


def test_an_allowed_swi_builtin_needs_no_registration():
    prover = JanusProver(HERE / "toy_engine.pl")

    assert _answers(prover, FACTS, "length([a,b], N)") == {
        parse_symbol("length([a,b], 2)")
    }


def test_a_builtin_belongs_to_the_prover_that_added_it():
    def square(lhs, rhs):
        yield {rhs: (str(int(lhs[0]) ** 2),)}

    with_square = JanusProver(HERE / "toy_engine.pl")
    without = JanusProver(HERE / "toy_engine.pl")
    with_square.add_builtin("square", 2, square)

    assert _answers(with_square, FACTS, "square(3, Y)") == {
        parse_symbol("square(3, 9)")
    }
    with pytest.raises(UnknownPredicateException):
        _answers(without, FACTS, "square(3, Y)")


def test_engines_defining_the_same_names_stay_apart():
    toy = JanusProver(HERE / "toy_engine.pl")
    other = JanusProver(HERE / "other_engine.pl")

    assert _answers(other, FACTS, "p(X)") == {("other",)}
    assert parse_symbol("p(a)") in _answers(toy, FACTS, "p(X)")


def test_the_library_exports_its_documented_contract():
    from janus_swi import janus

    JanusProver(HERE / "toy_engine.pl")
    exports = janus.query_once(
        "findall(_S, (module_property(deeplog_grounding, exports(_E)),"
        ' member(_N/_A, _E), format(string(_S), "~w/~w", [_N, _A])), Exports)'
    )["Exports"]
    documented = re.findall(r"^\s*\* - ``(\w+/\d+)``", DOCS.read_text(), re.MULTILINE)

    assert set(exports) == EXPORTS
    assert set(documented) == EXPORTS

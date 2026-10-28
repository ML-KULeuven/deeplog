#  Copyright (c) 2024-2026. KU Leuven
"""What grounding through Janus leaves behind in Prolog."""

import pytest


pytest.importorskip("janus_swi")

import janus_swi as janus

from deeplog.grounding.prolog import JanusGrounder
from deeplog.grounding.prolog import UnknownPredicateException
from deeplog.grounding.prolog import str_to_rules


#: The sum of two digits, which has 19 answers over 100 proofs. Its tables take
#: some 200 kB.
ADDITION = (
    "addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), "
    "digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).\n"
    "digit(I,N).\n"
    "?- addition(i1,i2,S)."
)


def _table_space() -> int:
    """The bytes the calling Prolog thread's tables take, once what was dropped is freed."""
    return janus.query_once(
        "garbage_collect_atoms, statistics(table_space_used, Used)"
    )["Used"]


def _prove(grounder: JanusGrounder, code: str):
    program = tuple(str_to_rules(code))
    return grounder.ground(program, program[-1][2], {("digit", 2)})


#: What a proof may leave in the table space: far less than a table.
TRACE = 1024


def test_a_proof_leaves_no_table_behind():
    """Proving a goal again takes the table space of proving it once.

    Each proof tables its goals under a builder of its own, so a table that
    stayed would be read by no later proof, and would hold what that proof
    built.
    """
    grounder = JanusGrounder()

    first = _prove(grounder, ADDITION)
    after_one = _table_space()
    again = _prove(grounder, ADDITION)

    assert len(first) == 19
    assert first.keys() == again.keys()
    assert _table_space() - after_one < TRACE


def test_a_proof_that_fails_leaves_no_table_behind():
    """An error, here after the digits are proved, ends a proof as its answers do."""
    grounder = JanusGrounder()
    _prove(grounder, ADDITION)
    before = _table_space()

    with pytest.raises(UnknownPredicateException):
        _prove(grounder, ADDITION.replace("is(S,+(N1,N2)).", "is(S,+(N1,N2)), odd(S)."))

    assert _table_space() - before < TRACE

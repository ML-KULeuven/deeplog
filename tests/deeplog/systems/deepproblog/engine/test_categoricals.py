#  Copyright (c) 2024-2026. KU Leuven
"""Tests for annotated disjunctions (categoricals).

Tests cover three layers:
- Structural — parsing AD-fact rewrites, per-proof mutex filtering
  in SimpleEngine and KBestJanusEngine.
- Engine parity — all three engines produce the same set of ground
  goals for an AD program.
- End-to-end probability — compile through MV-SDD and verify each
  engine yields the correct marginal probability, including
  JanusEngine whose tabled prover doesn't filter conflicting proofs
  at construction time but whose formulas are mutex-corrected by
  MV-SDD canonicalisation.
"""

from typing import cast

import pytest
import torch

from deeplog.circuit import to_module as circuit_to_module
from deeplog.formula import DeepLogModuleFactory
from deeplog.formula import SymbolicFormulaFactory
from deeplog.symbol import Symbol
from deeplog.systems.deepproblog.engine import Engine
from deeplog.systems.deepproblog.engine import JanusEngine
from deeplog.systems.deepproblog.engine import KBestJanusEngine
from deeplog.systems.deepproblog.engine import SimpleEngine
from deeplog.systems.deepproblog.program import expand_annotated_disjunctions
from deeplog.systems.deepproblog.program import is_categorical_label
from deeplog.systems.deepproblog.program import str_to_rule
from deeplog.systems.deepproblog.program import str_to_rules
from deeplog.systems.deepproblog.program import unwrap_categorical_label


pymvsdd = pytest.importorskip("pymvsdd")


# Engines that enforce per-proof mutual exclusivity by enumerating proofs.
mutex_engines: list[type[Engine]] = [SimpleEngine]
if KBestJanusEngine.is_available():
    mutex_engines.append(KBestJanusEngine)


# All engines that should at least round-trip an AD program without crashing.
all_engines: list[type[Engine]] = [SimpleEngine]
if JanusEngine.is_available():
    all_engines.append(JanusEngine)
if KBestJanusEngine.is_available():
    all_engines.append(KBestJanusEngine)


def _make_engine(engine_class: type[Engine]) -> Engine:
    """Instantiate ``engine_class`` with whatever constructor args it needs."""
    if engine_class is KBestJanusEngine:
        return KBestJanusEngine(k=10)
    return engine_class()


def test_ad_parses_to_branches():
    """`p1::a; p2::b.` expands to N regular facts with categorical-wrapped labels."""
    rules = [str_to_rule("0.3::a; 0.7::b.")]
    expanded = list(expand_annotated_disjunctions(rules))
    assert len(expanded) == 2
    fact_a, fact_b = expanded
    # Both facts share the same cat_id; value indices are 0 and 1.
    label_a = cast(Symbol, fact_a[1][1])  # ('::', label, atom) — pick label
    label_b = cast(Symbol, fact_b[1][1])
    assert is_categorical_label(label_a)
    assert is_categorical_label(label_b)
    inner_a, cat_a, idx_a = unwrap_categorical_label(label_a)
    inner_b, cat_b, idx_b = unwrap_categorical_label(label_b)
    assert inner_a == ("0.3",)
    assert inner_b == ("0.7",)
    assert cat_a == cat_b
    assert (idx_a, idx_b) == (0, 1)


def test_independent_ads_get_different_cat_ids():
    rules = list(str_to_rules("0.3::a; 0.7::b.\n0.4::c; 0.6::d."))
    expanded = list(expand_annotated_disjunctions(rules))
    assert len(expanded) == 4
    cat_ids = {
        unwrap_categorical_label(cast(Symbol, rule[1][1]))[1] for rule in expanded
    }
    assert len(cat_ids) == 2  # one per AD


@pytest.mark.parametrize("engine_class", all_engines)
def test_simple_ad_query(engine_class: type[Engine]):
    """A query against one branch of an AD returns that branch's leaf."""
    engine = _make_engine(engine_class)
    program = tuple(str_to_rules("0.3::a; 0.7::b.\n?- a."))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    assert ("a",) in result.formulas
    # The factory sees the unwrapped (inner) label, not the categorical wrapping.
    assert result.labels[("a",)] == ("0.3",)


@pytest.mark.parametrize("engine_class", mutex_engines)
def test_ad_mutex_within_proof(engine_class: type[Engine]):
    """`?- a, b.` over `0.3::a; 0.7::b.` must not produce a proof.

    JanusEngine is excluded — its tabled prover doesn't filter at proof
    construction time; downstream MV-SDD compilation is responsible
    there.
    """
    engine = _make_engine(engine_class)
    program = tuple(str_to_rules("0.3::a; 0.7::b.\n?- a, b."))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    # The conflicting proof was dropped → no ground goal for `a, b`.
    assert (",", ("a",), ("b",)) not in result.formulas
    assert result.formulas == {}


@pytest.mark.parametrize("engine_class", all_engines)
def test_ad_in_rule_body(engine_class: type[Engine]):
    engine = _make_engine(engine_class)
    code = """
    0.3::heads; 0.7::tails.
    happy :- heads.
    ?- happy.
    """
    program = tuple(str_to_rules(code))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    assert ("happy",) in result.formulas
    assert result.labels[("heads",)] == ("0.3",)


@pytest.mark.parametrize("engine_class", mutex_engines)
def test_multiple_independent_ads(engine_class: type[Engine]):
    """Different ADs are independent: a proof using one branch from each is fine."""
    engine = _make_engine(engine_class)
    code = """
    0.3::a; 0.7::b.
    0.4::c; 0.6::d.
    ?- a, c.
    """
    program = tuple(str_to_rules(code))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    assert (",", ("a",), ("c",)) in result.formulas


@pytest.mark.parametrize("engine_class", mutex_engines)
def test_ad_branches_are_independent_proofs(engine_class: type[Engine]):
    """A variable query over an AD enumerates all branches as separate proofs."""
    engine = _make_engine(engine_class)
    code = "0.2::p(a); 0.5::p(b); 0.3::p(c).\n?- p(X)."
    program = tuple(str_to_rules(code))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    assert set(result.formulas.keys()) == {
        ("p", ("a",)),
        ("p", ("b",)),
        ("p", ("c",)),
    }


def test_ad_engine_parity():
    """SimpleEngine, JanusEngine, KBestJanusEngine return the same ground goals."""
    if not (JanusEngine.is_available() and KBestJanusEngine.is_available()):
        pytest.skip("janus_swi not available")
    code = """
    0.3::a; 0.7::b.
    0.4::c; 0.6::d.
    happy :- a, c.
    happy :- b, d.
    ?- happy.
    """
    program = tuple(str_to_rules(code))
    simple = SimpleEngine().get_query_result(program, SymbolicFormulaFactory())
    janus = JanusEngine().get_query_result(program, SymbolicFormulaFactory())
    kbest = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    assert set(simple.formulas) == set(janus.formulas) == set(kbest.formulas)


# --- Probability tests via MV-SDD compilation -----------------------------
# These exercise the downstream MV-SDD circuit backend. With probability
# semiring (sum/product over real), each query's output equals its
# declared marginal under one-hot AD inputs.


def _compile_for_probability(code: str):
    """Compile an AD program through MV-SDD with the probability semiring."""
    program = tuple(str_to_rules(code))
    factory = DeepLogModuleFactory()
    result = SimpleEngine().get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    mod = circuit_to_module(
        *(n.root for n in nodes),
        names=answers,
        categoricals=result.categoricals,
        labels=result.labels,
        structure_override="probability",
    )
    return mod, answers


def test_ad_branch_probabilities_sum_to_one():
    """For a sum-to-one AD with constant labels, the AC bakes the
    declared probabilities in: it takes no runtime input and outputs
    each branch's marginal directly. The sum across branches is 1."""
    mod, answers = _compile_for_probability(
        """
        0.3::a; 0.5::b; 0.2::c.
        ?- a.
        ?- b.
        ?- c.
        """
    )
    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    out_by_query = dict(zip(answers, out[0].tolist(), strict=True))
    assert pytest.approx(out_by_query[("a",)], abs=1e-5) == 0.3
    assert pytest.approx(out_by_query[("b",)], abs=1e-5) == 0.5
    assert pytest.approx(out_by_query[("c",)], abs=1e-5) == 0.2
    assert pytest.approx(sum(out_by_query.values()), abs=1e-5) == 1.0


def test_ad_residual_branch_probability():
    """`0.3::a; 0.5::b.` (sum < 1): the residual 0.2 is the implicit
    "neither branch chosen" outcome. With constant labels both queries
    are baked into the AC, so the two output marginals are 0.3 and
    0.5 — they sum to 0.8 with no implicit renormalisation."""
    mod, answers = _compile_for_probability(
        """
        0.3::a; 0.5::b.
        ?- a.
        ?- b.
        """
    )
    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    out_by_query = dict(zip(answers, out[0].tolist(), strict=True))
    assert pytest.approx(out_by_query[("a",)], abs=1e-5) == 0.3
    assert pytest.approx(out_by_query[("b",)], abs=1e-5) == 0.5
    assert pytest.approx(sum(out_by_query.values()), abs=1e-5) == 0.8


# --- Engine parity through MV-SDD -----------------------------------------
# The boundary that matters for *probability* parity isn't the engine's
# raw formula — it's the MV-SDD-compiled AC. JanusEngine's tabled
# prover doesn't enforce per-proof mutex (a query that conjoins two
# branches of one AD produces a literal-conjoined formula at engine
# level), but MV-SDD canonicalisation reduces such conjunctions to ⊥
# at compile time. So all three engines should land on the same
# numerical outputs once compiled.


_PARITY_PROBABILITY_PROGRAM = """
0.3::a; 0.5::b; 0.2::c.
0.4::x; 0.6::y.
joint(a, x) :- a, x.
joint(a, y) :- a, y.
joint(b, x) :- b, x.
joint(b, y) :- b, y.
joint(c, x) :- c, x.
joint(c, y) :- c, y.
?- joint(N1, N2).
"""


def _compile_with_engine(engine: Engine, code: str):
    program = tuple(str_to_rules(code))
    factory = DeepLogModuleFactory()
    result = engine.get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    mod = circuit_to_module(
        *(n.root for n in nodes),
        names=answers,
        categoricals=result.categoricals,
        labels=result.labels,
        structure_override="probability",
    )
    out = mod(torch.zeros((1, 0)))
    return dict(zip(answers, out[0].tolist(), strict=True))


@pytest.mark.parametrize(
    "engine_class",
    [
        SimpleEngine,
        pytest.param(
            JanusEngine,
            marks=pytest.mark.skipif(
                not JanusEngine.is_available(), reason="janus_swi not available"
            ),
        ),
        pytest.param(
            KBestJanusEngine,
            marks=pytest.mark.skipif(
                not KBestJanusEngine.is_available(),
                reason="janus_swi not available",
            ),
        ),
    ],
)
def test_engine_marginals_through_mvsdd(engine_class):
    # All engines (including the tabled, mutex-unfiltered JanusEngine)
    # land on the correct joint probabilities once compiled to MV-SDD.
    # P(joint(n1, n2)) = P_cat0(n1) · P_cat1(n2) for independent ADs.
    engine = engine_class(k=20) if engine_class is KBestJanusEngine else engine_class()
    out_by_query = _compile_with_engine(engine, _PARITY_PROBABILITY_PROGRAM)

    p_cat0 = {"a": 0.3, "b": 0.5, "c": 0.2}
    p_cat1 = {"x": 0.4, "y": 0.6}
    for n1, p1 in p_cat0.items():
        for n2, p2 in p_cat1.items():
            key = ("joint", (n1,), (n2,))
            assert pytest.approx(out_by_query[key], abs=1e-5) == p1 * p2


@pytest.mark.skipif(not JanusEngine.is_available(), reason="janus_swi not available")
def test_janus_within_ad_conjunction_zeroed_by_mvsdd():
    # JanusEngine emits the formula (leaf_a ∧ leaf_b) for `?- a, b.`
    # because tabling skips per-proof mutex checks. MV-SDD compilation
    # reduces [var=0] ∧ [var=1] to ⊥, so the AC outputs 0 — matching
    # what SimpleEngine/KBestJanusEngine produce by dropping the
    # conflicting proof at construction time.
    code = """
    0.3::a; 0.7::b.
    impossible :- a, b.
    ?- impossible.
    """
    program = tuple(str_to_rules(code))
    factory = DeepLogModuleFactory()
    result = JanusEngine().get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    mod = circuit_to_module(
        *(n.root for n in nodes),
        names=answers,
        categoricals=result.categoricals,
        labels=result.labels,
        structure_override="probability",
    )
    out = mod(torch.zeros((1, 0)))
    assert pytest.approx(float(out[0, 0]), abs=1e-5) == 0.0

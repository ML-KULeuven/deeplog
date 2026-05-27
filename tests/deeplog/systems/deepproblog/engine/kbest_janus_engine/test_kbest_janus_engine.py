#  Copyright (c) 2024-2026. KU Leuven
"""Tests for :class:`KBestJanusEngine`."""

import pytest
import torch

from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula.predicates.builtin_predicates import get_network_predicate
from deeplog.systems.deepproblog.engine import JanusEngine
from deeplog.systems.deepproblog.engine import KBestJanusEngine
from deeplog.systems.deepproblog.engine import UnknownPredicateException
from deeplog.systems.deepproblog.engine.engine import EngineFactory
from deeplog.systems.deepproblog.engine.kbest_janus_engine.predicate_evaluator import (
    NeuralPredicateEvaluator,
)
from deeplog.systems.deepproblog.program import str_to_rules


pytestmark = pytest.mark.skipif(
    not KBestJanusEngine.is_available(),
    reason="janus_swi not available",
)


PROGRAM_THREE_FACTS = """
0.8::a.
0.5::b.
0.2::c.
q :- a.
q :- b.
q :- c.
?-q.
"""


def _single_formula(result):
    assert len(result.formulas) == 1
    return next(iter(result.formulas.values()))


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        KBestJanusEngine(k=0)


def test_heuristic_must_be_known():
    with pytest.raises(ValueError):
        KBestJanusEngine(k=1, heuristic="not-a-heuristic")


def test_k1_picks_highest_probability():
    program = tuple(str_to_rules(PROGRAM_THREE_FACTS))
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "a_boolean"


def test_k2_picks_top_two():
    program = tuple(str_to_rules(PROGRAM_THREE_FACTS))
    result = KBestJanusEngine(k=2).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "a_boolean or b_boolean"


def test_kge_all_matches_full_enumeration():
    program = tuple(str_to_rules(PROGRAM_THREE_FACTS))
    kbest = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    janus = JanusEngine().get_query_result(program, SymbolicFormulaFactory())
    # Same set of ground goals, same disjunctive content (order may differ
    # because JanusEngine's tabled disjoin is bottom-up while KBest folds
    # top-down by probability). Compare by splitting on "or".
    assert set(kbest.formulas.keys()) == set(janus.formulas.keys())
    for goal in kbest.formulas:
        kbest_terms = sorted(
            str(kbest.formulas[goal]).replace("(", "").replace(")", "").split(" or ")
        )
        janus_terms = sorted(
            str(janus.formulas[goal]).replace("(", "").replace(")", "").split(" or ")
        )
        assert kbest_terms == janus_terms


def test_conjunction_in_rule_body():
    code = """
    0.9::a.
    0.1::b.
    0.5::c.
    q :- a, b.
    q :- c.
    ?-q.
    """
    program = tuple(str_to_rules(code))
    # Top proof: c (P=0.5). Second: a,b (P=0.09). k=1 should keep only c.
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "c_boolean"
    # k=2 should include both proofs.
    result = KBestJanusEngine(k=2).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "c_boolean or (a_boolean and b_boolean)"


def test_unknown_predicate_raises():
    code = "?-missing."
    program = tuple(str_to_rules(code))
    with pytest.raises(UnknownPredicateException):
        KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())


def test_negation_hoists_single_rv():
    # `not(a)` over a single Boolean RV must split the world: the surviving
    # branch (a=false) carries the negated leaf with probability 1-P(a)=0.7.
    code = """0.3::a.
?-not(a)."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "not a_boolean"


def test_negation_over_deterministic_goal_no_split():
    # NAF over a defined-but-failing predicate is fully deterministic:
    # the subproof fails without touching any probabilistic fact, so
    # no choice is hoisted and no leaf is added to the formula.
    code = """0.5::a.
unsat :- between(1, 0, _).
q :- not(unsat), a.
?-q."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "a_boolean"


def test_negation_respects_prior_commitment():
    # `a` is consumed positively before `not(a)` is evaluated, so the NAF
    # subproof reads a=true from the per-branch assignment, immediately
    # fails the branch — and the only top-K result must come from the
    # other rule (`q :- not(a)` with no positive `a` consumption first,
    # which hoists and keeps the a=false branch).
    code = """
    0.4::a.
    q :- a, not(a).
    q :- not(a).
    ?-q.
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    # Only the second rule should survive; its formula is the negated leaf.
    assert _single_formula(result) == "not a_boolean"


def test_consistency_naf_then_positive_same_rv():
    # `q :- not(a), a.` is unsatisfiable: NAF commits a=false, then the
    # positive `a` goal needs a=true, which contradicts the assignment.
    # The fact-expand clause must reject the inconsistent commitment so
    # this branch dies; otherwise we'd produce a bogus proof with
    # formula `not(a) AND a` and non-zero probability mass.
    code = """0.5::a.
q :- not(a), a.
?-q."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    assert result.formulas == {}


def test_consistency_positive_then_naf_same_rv():
    # Mirror direction: `q :- a, not(a).` — positive commit first, then
    # NAF reads a=true and immediately fails the branch. Already handled
    # correctly by `naf_resolve` consulting the assignment.
    code = """0.5::a.
q :- a, not(a).
?-q."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    assert result.formulas == {}


def test_double_negation_equals_positive():
    # `not(not(a))` should be probability-equivalent to `a`. The throw
    # from the inner `naf_resolve(a)` must propagate cleanly through
    # the recursive `\\+ naf_resolve(...)` clause used for nested NAF.
    code = """0.3::a.
?-not(not(a))."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "a_boolean"


def test_triple_negation_equals_single_negation():
    code = """0.3::a.
?-not(not(not(a)))."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "not a_boolean"


def test_negation_over_deterministically_succeeding_predicate_kills_branch():
    # `not(g)` where `g` succeeds deterministically (no facts, just a
    # rule body that always holds) must fail the branch — no proofs.
    code = """0.5::a.
g :- between(0, 0, _).
q :- not(g), a.
?-q."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert result.formulas == {}


def test_negation_over_unknown_predicate_raises():
    # Unknown predicate inside NAF should raise UnknownPredicateException
    # for parity with the main engine. Currently `naf_resolve` falls
    # through all clauses and silently fails, making `not(missing)`
    # succeed — that's the asymmetry this test pins down.
    code = "?-not(missing)."
    program = tuple(str_to_rules(code))
    with pytest.raises(UnknownPredicateException):
        KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())


def test_negation_of_disjunction():
    # `not(a;b)` keeps only the all-false world: P = (1-Pa)(1-Pb).
    # Wrapped in two rules because str_to_rules doesn't keep `;` inside
    # parens of a top-level NAF.
    code = """0.5::a.
0.5::b.
ab :- a.
ab :- b.
?-not(ab)."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    formula = str(_single_formula(result))
    terms = sorted(formula.replace("(", "").replace(")", "").split(" and "))
    assert terms == sorted(["not a_boolean", "not b_boolean"])


def test_negation_of_conjunction():
    # `not(a,b)` survives in any world where a or b is false.
    # Surviving worlds: a=false (b unconstrained, P=0.5), and a=true,b=false
    # (P=0.25). Total P=0.75. Wrapped in a rule because str_to_rules
    # doesn't keep `,` inside parens of a top-level NAF either.
    code = """0.5::a.
0.5::b.
ab :- a, b.
?-not(ab)."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    formula = str(_single_formula(result))
    terms = sorted(formula.replace("(", "").replace(")", "").split(" or "))
    assert terms == sorted(["not a_boolean", "a_boolean and not b_boolean"])


def test_negation_over_existential():
    # NAF over an existential goal must hoist every ground RV the
    # subproof can touch. Equivalent to `not(p(X))`, expressed via a
    # wrapper rule so both engines produce a ground top-level query
    # symbol that compares cleanly.
    code = """0.5::p(1).
0.5::p(2).
some_p :- p(_).
?-not(some_p)."""
    program = tuple(str_to_rules(code))
    kbest = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    janus = JanusEngine().get_query_result(program, SymbolicFormulaFactory())
    assert set(kbest.formulas.keys()) == set(janus.formulas.keys())


def test_negation_only_proof_kept_at_low_probability():
    # Top-1 with k=1 must return the single surviving NAF branch even
    # when its post-NAF probability (1-P=0.1) is low — there is no
    # competing proof, so the heap must still emit it.
    code = """0.9::a.
?-not(a)."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "not a_boolean"


def test_repeated_naf_shares_assignment():
    # Two NAFs over the same RV in the same conjunction must share the
    # commitment from the first NAF's hoist — the second must NOT
    # re-hoist. Surviving branch has only one negated leaf in its formula.
    code = """0.5::a.
q :- not(a), not(a).
?-q."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    assert len(result.formulas) == 1
    formula = str(_single_formula(result))
    # The negated leaf is the only thing referencing a; no positive leaf.
    assert "not a_boolean" in formula
    stripped = formula.replace("not a_boolean", "")
    assert "a_boolean" not in stripped


def test_naf_and_positive_decompose_world():
    # `q :- a. q :- not(a).` partitions the world; together the two
    # proofs cover P=1 (one for each truth value of a).
    code = """0.4::a.
q :- a.
q :- not(a).
?-q."""
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    formula = str(_single_formula(result))
    terms = sorted(formula.replace("(", "").replace(")", "").split(" or "))
    assert terms == sorted(["a_boolean", "not a_boolean"])


def test_negation_top_k_matches_full_enumeration():
    # Same content as JanusEngine on a program with NAF, modulo disjunction
    # ordering (compare as sets of conjuncts).
    code = """
    0.6::a.
    0.5::b.
    q :- a, not(b).
    q :- not(a), b.
    ?-q.
    """
    program = tuple(str_to_rules(code))
    kbest = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    janus = JanusEngine().get_query_result(program, SymbolicFormulaFactory())
    assert set(kbest.formulas.keys()) == set(janus.formulas.keys())
    for goal in kbest.formulas:
        kbest_terms = sorted(
            str(kbest.formulas[goal]).replace("(", "").replace(")", "").split(" or ")
        )
        janus_terms = sorted(
            str(janus.formulas[goal]).replace("(", "").replace(")", "").split(" or ")
        )
        assert kbest_terms == janus_terms


def test_between_binds_variable():
    # `between/3` is a Prolog builtin that enumerates integer values; the
    # engine must treat it as a pure builtin, binding X across the rule body
    # and producing one ground output per value.
    code = """
    0.5::p.
    output(X) :- between(0,2,X), p.
    ?-output(X).
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=5).get_query_result(program, SymbolicFormulaFactory())
    assert set(result.formulas.keys()) == {
        ("output", ("0",)),
        ("output", ("1",)),
        ("output", ("2",)),
    }


def test_geometric_mean_ordering():
    # Both proofs have the same *total* probability (0.5 for the single fact,
    # 0.5*0.9 ≈ 0.45 for the two-fact chain). Under pp the single-fact proof
    # (P=0.5) wins at k=1. Under gm the two-fact proof has gm = (log 2 +
    # log(1/0.9))/2 ≈ 0.40; the single-fact proof has gm = log 2 ≈ 0.69 — so
    # gm prefers the chain.
    code = """
    0.5::p.
    0.5::q.
    0.9::r.
    goal :- p.
    goal :- q, r.
    ?-goal.
    """
    program = tuple(str_to_rules(code))
    pp = KBestJanusEngine(k=1, heuristic="pp").get_query_result(
        program, SymbolicFormulaFactory()
    )
    gm = KBestJanusEngine(k=1, heuristic="gm").get_query_result(
        program, SymbolicFormulaFactory()
    )
    assert _single_formula(pp) == "p_boolean"
    assert _single_formula(gm) == "q_boolean and r_boolean"


def test_engine_factory_scalar_probability_for_numeric_label():
    factory = EngineFactory(SymbolicFormulaFactory())
    assert factory.get_scalar_probability(("0.25",)) == pytest.approx(0.25)


def test_engine_factory_scalar_probability_raises_on_unresolvable_label():
    # Heuristic ranking requires a probability; a label that's neither a
    # numeric constant nor resolvable by the wrapped factory is a bug, not
    # a neutral case.
    factory = EngineFactory(SymbolicFormulaFactory())
    with pytest.raises(ValueError):
        factory.get_scalar_probability(("tag",))


def test_map_inference_returns_best_proof():
    # MAP inference on the classic wet-grass example. `wet` has two competing
    # proofs; the single highest-probability one (the MAP explanation) is the
    # sprinkler branch, even though rain's marginal probability is lower than
    # sprinkler's only by a little:
    #   rain branch:      P(rain) * P(wet_from_rain)           = 0.6 * 0.8 = 0.48
    #   sprinkler branch: P(sprinkler) * P(wet_from_sprinkler) = 0.9 * 0.7 = 0.63
    code = """
    0.6::rain.
    0.9::sprinkler.
    0.8::wet_from_rain.
    0.7::wet_from_sprinkler.
    wet :- rain, wet_from_rain.
    wet :- sprinkler, wet_from_sprinkler.
    ?-wet.
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=1).get_query_result(program, SymbolicFormulaFactory())
    assert _single_formula(result) == "sprinkler_boolean and wet_from_sprinkler_boolean"
    # The MAP explanation uses two facts; its probability is the product of
    # their labels and must equal the argmax over both branches.
    labels = result.labels
    map_probability = float(labels[("sprinkler",)][0]) * float(
        labels[("wet_from_sprinkler",)][0]
    )
    assert map_probability == pytest.approx(0.63)


class _DummyClassifier(torch.nn.Module):
    def forward(self, x):
        return x


def test_categorical_with_tensor_input():
    # Neural-categorical AD: the per-class probability comes from a registered
    # network predicate. KBest stays in symbolic-formula land (boolean leaves
    # keyed on goals); the neural label probability is supplied via a
    # ``NeuralPredicateEvaluator`` consumed only by the ranking heuristic.
    code = (
        "classifier(img1, 0) :: class(img1, knight); "
        "classifier(img1, 1) :: class(img1, rook); "
        "classifier(img1, 2) :: class(img1, bishop); "
        "classifier(img1, 3) :: class(img1, king); "
        "classifier(img1, 4) :: class(img1, queen).\n"
        "?-class(img1, Class).\n"
    )
    evaluator = NeuralPredicateEvaluator(
        tensors={("img1",): torch.tensor([0.1, 0.6, 0.1, 0.1, 0.1])},
        atom_builders={
            ("classifier", 2, "probability"): get_network_predicate(
                "classifier", 2, "probability", _DummyClassifier()
            ),
        },
    )
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=1).get_query_result(
        program, SymbolicFormulaFactory(), evaluator=evaluator
    )
    # Top-1 proof: rook (the index-1 entry, score 0.6) — selection alone
    # demonstrates the heuristic correctly read the network output. The
    # leaf is keyed directly on the AD branch's head atom; the network
    # probability lives in the labels mapping (unwrapped from the
    # categorical wrapping by the engine before reaching the factory).
    assert set(result.formulas.keys()) == {("class", ("img1",), ("rook",))}
    assert (
        result.formulas[("class", ("img1",), ("rook",))] == "class(img1,rook)_boolean"
    )
    assert result.labels[("class", ("img1",), ("rook",))] == (
        "classifier",
        ("img1",),
        ("1",),
    )
    # Engine reports the leaf's (cat_id, value_idx) tagging so downstream
    # MV-SDD compilation can fan a single multi-valued literal at this AD.
    cat_id, value_idx = result.categoricals[("class", ("img1",), ("rook",))]
    assert value_idx == 1
    assert cat_id.startswith("@cat_id_")
    # The evaluator is independently sanity-checked: classifier(img1, 1)
    # must give the rook's probability.
    assert evaluator(("classifier", ("img1",), ("1",))) == pytest.approx(0.6)


def test_naf_over_categorical_two_branch():
    # `?- not(a).` over `0.3::a; 0.7::b.` triggers the categorical hoist:
    # 3 heap children spawn (a-pos, b-pos, none); the a-pos child fails
    # itself because committing `a` makes `not(a)` false. The two
    # surviving children produce: leaf(b) and the single "RV took none"
    # MV literal.
    code = """
    0.3::a; 0.7::b.
    ?- not(a).
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    assert set(result.formulas.keys()) == {("not", ("a",))}
    formula_str = str(result.formulas[("not", ("a",))])
    # b-positive child survives — its MV literal is a regular indicator
    # tagged ``_boolean``; the AD-specific (cat_id, value) info lives in
    # ``result.categoricals``, not in the leaf string.
    assert "b_boolean" in formula_str
    # "none" child survives — single MV literal for the residual outcome,
    # named ``@cat_none(@cat_id_N)``.
    assert "@cat_none" in formula_str


def test_naf_over_categorical_three_branch():
    # 3-branch AD: 4 heap children spawn (a, b, c, none); only a-pos
    # fails on re-resolution. The surviving disjunction includes b, c,
    # and the single "none"-outcome MV literal.
    code = """
    0.2::a; 0.5::b; 0.3::c.
    ?- not(a).
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    formula_str = str(result.formulas[("not", ("a",))])
    assert "b_boolean" in formula_str
    assert "c_boolean" in formula_str
    # The "none" outcome is now a single MV literal, not a conjunction
    # of negated branches.
    assert "@cat_none" in formula_str
    # And the engine reports the (cat_id, "none") tagging for it.
    none_tags = {v for v in result.categoricals.values() if v[1] == "none"}
    assert len(none_tags) == 1


def test_naf_in_rule_body_with_positive_ad_branch():
    # `goal :- not(a), b.` over the AD: only the b-pos child survives.
    # In the a-pos child, not(a) fails (committed → Goal true). In the
    # none child, the subsequent `b` consumption fails (cat is committed
    # to "none", different from the b branch's value-idx). Net result:
    # one ground proof, formula = the single MV literal for b.
    code = """
    0.3::a; 0.7::b.
    goal :- not(a), b.
    ?- goal.
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    assert set(result.formulas.keys()) == {("goal",)}
    assert str(result.formulas[("goal",)]) == "b_boolean"


def test_naf_over_non_categorical_still_works():
    # Sanity check: NAF over a regular probabilistic fact still hoists
    # correctly and produces a negated leaf — the categorical guard
    # doesn't accidentally block the boolean-NAF path added previously.
    code = """
    0.4::p.
    q :- not(p).
    ?- q.
    """
    program = tuple(str_to_rules(code))
    result = KBestJanusEngine(k=10).get_query_result(program, SymbolicFormulaFactory())
    assert ("q",) in result.formulas
    assert "not p_boolean" in str(result.formulas[("q",)])

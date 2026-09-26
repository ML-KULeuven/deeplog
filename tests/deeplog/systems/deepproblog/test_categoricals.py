#  Copyright (c) 2024-2026. KU Leuven
"""Tests for annotated disjunctions (categorical variables).

Tests cover four layers:
- Structural — parsing AD-fact rewrites, plus per-proof mutex filtering
  which (after the grounder split) only the k-best prover still does at
  construction time.
- Recognition — the base path keeps the disjunction whole and reads the
  variable back off the ground atoms, so a variable's domain holds values
  and the atoms asserting them are recognition's output.
- Grounder parity — all three grounders produce the same set of ground
  goals for an AD program.
- End-to-end probability — compile through MV-SDD and verify each
  grounder yields the correct marginal probability. SimpleGrounder and
  JanusGrounder no longer filter conflicting proofs at construction; their
  formulas are mutex-corrected by MV-SDD canonicalisation instead.
"""

from typing import cast

import pytest
import torch

from deeplog import to_dict
from deeplog import to_module as circuit_to_module
from deeplog.formula import DeepLogModuleFactory
from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.formula.predicates.builtin_predicates import get_network_predicate
from deeplog.formula.strategies import transform_expectation_to_probability
from deeplog.grounding import JanusGrounder
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import str_to_rule
from deeplog.grounding import str_to_rules
from deeplog.shape import get_all_symbols
from deeplog.symbol import Symbol
from deeplog.symbol import with_structure
from deeplog.systems.deepproblog import KBestJanusGrounder
from deeplog.systems.deepproblog import Solver
from deeplog.systems.deepproblog import compile_to_module
from deeplog.systems.deepproblog.compile import build_leaf_mapping
from deeplog.systems.deepproblog.kbest.kbest import expand_annotated_disjunctions
from deeplog.util import as_tuple
from deeplog.variable import OPEN
from deeplog.variable import atom_asserting
from deeplog.variable import indicated_values


pymvsdd = pytest.importorskip("pymvsdd")


# Grounders that enforce per-proof mutual exclusivity during search. Only the
# k-best prover still filters conflicting AD branches at construction time;
# SimpleGrounder now defers mutex to MV-SDD compilation (matching JanusGrounder).
mutex_engines: list = []
if KBestJanusGrounder.is_available():
    mutex_engines.append(KBestJanusGrounder)


# All grounders that should at least round-trip an AD program without crashing.
all_engines: list = [SimpleGrounder]
if JanusGrounder.is_available():
    all_engines.append(JanusGrounder)
if KBestJanusGrounder.is_available():
    all_engines.append(KBestJanusGrounder)


def _make_engine(engine_class):
    """Build a solver/grounder for ``engine_class`` with the args it needs."""
    if engine_class is KBestJanusGrounder:
        return KBestJanusGrounder(k=10)
    return Solver(engine_class())


def test_ad_parses_to_branches():
    """The k-best rendering expands `p1::a; p2::b.` into N `@cat`-tagged facts.

    Only k-best needs the tag: its `engine.pl` matches on it to recognize a
    branch *during* search. The base grounder keeps the disjunction whole and
    recognizes it afterwards — see the recognition tests at the end of this file.
    """
    rules = [str_to_rule("0.3::a; 0.7::b.")]
    expanded = list(expand_annotated_disjunctions(rules))
    assert len(expanded) == 2
    fact_a, fact_b = expanded
    # ('::', label, atom) — pick the label. Both branches share a cat-id and
    # differ in value index; ``engine.pl`` matches on exactly this shape.
    assert fact_a[1][1] == ("@cat", ("0.3",), ("@cat_id_0",), ("0",))
    assert fact_b[1][1] == ("@cat", ("0.7",), ("@cat_id_0",), ("1",))


def test_independent_ads_get_different_cat_ids():
    rules = list(str_to_rules("0.3::a; 0.7::b.\n0.4::c; 0.6::d."))
    expanded = list(expand_annotated_disjunctions(rules))
    assert len(expanded) == 4
    cat_ids = {cast(Symbol, rule[1][1])[2] for rule in expanded}
    assert len(cat_ids) == 2  # one per AD


@pytest.mark.parametrize("engine_class", all_engines)
def test_simple_ad_query(engine_class):
    """A query against one branch of an AD returns that branch's leaf."""
    engine = _make_engine(engine_class)
    program = tuple(str_to_rules("0.3::a; 0.7::b.\n?- a."))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    assert ("a",) in result.formulas
    # The factory sees the unwrapped (inner) label, not the categorical wrapping.
    assert result.labels[("a",)] == ("0.3",)


@pytest.mark.parametrize("engine_class", mutex_engines)
def test_ad_mutex_within_proof(engine_class):
    """`?- a, b.` over `0.3::a; 0.7::b.` must not produce a proof.

    Only k-best filters conflicting AD branches at proof-construction time.
    SimpleGrounder and JanusGrounder now emit the conjunction and rely on
    downstream MV-SDD compilation to zero it out (see the MV-SDD parity tests).
    """
    engine = _make_engine(engine_class)
    program = tuple(str_to_rules("0.3::a; 0.7::b.\n?- a, b."))
    result = engine.get_query_result(program, SymbolicFormulaFactory())
    # The conflicting proof was dropped → no ground goal for `a, b`.
    assert (",", ("a",), ("b",)) not in result.formulas
    assert result.formulas == {}


@pytest.mark.parametrize("engine_class", all_engines)
def test_ad_in_rule_body(engine_class):
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


@pytest.mark.parametrize("engine_class", all_engines)
def test_multiple_independent_ads(engine_class):
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


@pytest.mark.parametrize("engine_class", all_engines)
def test_ad_branches_are_independent_proofs(engine_class):
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
    """SimpleGrounder, JanusGrounder, KBestJanusGrounder return the same goals."""
    if not (JanusGrounder.is_available() and KBestJanusGrounder.is_available()):
        pytest.skip("janus_swi not available")
    code = """
    0.3::a; 0.7::b.
    0.4::c; 0.6::d.
    happy :- a, c.
    happy :- b, d.
    ?- happy.
    """
    program = tuple(str_to_rules(code))
    simple = Solver(SimpleGrounder()).get_query_result(
        program, SymbolicFormulaFactory()
    )
    janus = Solver(JanusGrounder()).get_query_result(program, SymbolicFormulaFactory())
    kbest = KBestJanusGrounder(k=10).get_query_result(program, SymbolicFormulaFactory())
    assert set(simple.formulas) == set(janus.formulas) == set(kbest.formulas)


# --- Probability tests via MV-SDD compilation -----------------------------
# These exercise the downstream MV-SDD circuit backend. With probability
# semiring (sum/product over real), each query's output equals its
# declared marginal under one-hot AD inputs.


def _compile_for_probability(code: str):
    """Knowledge-compile an AD program and count it in probability."""
    program = tuple(str_to_rules(code))
    factory = CircuitFactory()
    result = Solver(SimpleGrounder()).get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    mod = circuit_to_module(
        *transform_expectation_to_probability(
            *nodes,
            leaf_mapping=build_leaf_mapping(result.labels),
            variables=result.variables,
        ),
        names=answers,
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


# --- Grounder parity through MV-SDD ---------------------------------------
# The boundary that matters for *probability* parity isn't the grounder's
# raw formula — it's the MV-SDD-compiled AC. SimpleGrounder and JanusGrounder
# don't enforce per-proof mutex (a query that conjoins two branches of one AD
# produces a literal-conjoined formula at grounder level), but MV-SDD
# canonicalisation reduces such conjunctions to ⊥ at compile time. So all
# three grounders should land on the same numerical outputs once compiled.


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


def _compile_with_engine(engine, code: str):
    program = tuple(str_to_rules(code))
    factory = CircuitFactory()
    result = engine.get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    mod = circuit_to_module(
        *transform_expectation_to_probability(
            *nodes,
            leaf_mapping=build_leaf_mapping(result.labels),
            variables=result.variables,
        ),
        names=answers,
    )
    out = mod(torch.zeros((1, 0)))
    return dict(zip(answers, out[0].tolist(), strict=True))


@pytest.mark.parametrize(
    "engine_class",
    [
        SimpleGrounder,
        pytest.param(
            JanusGrounder,
            marks=pytest.mark.skipif(
                not JanusGrounder.is_available(), reason="janus_swi not available"
            ),
        ),
        pytest.param(
            KBestJanusGrounder,
            marks=pytest.mark.skipif(
                not KBestJanusGrounder.is_available(),
                reason="janus_swi not available",
            ),
        ),
    ],
)
def test_engine_marginals_through_mvsdd(engine_class):
    # All grounders (including the tabled, mutex-unfiltered JanusGrounder)
    # land on the correct joint probabilities once compiled to MV-SDD.
    # P(joint(n1, n2)) = P_cat0(n1) · P_cat1(n2) for independent ADs.
    engine = (
        KBestJanusGrounder(k=20)
        if engine_class is KBestJanusGrounder
        else Solver(engine_class())
    )
    out_by_query = _compile_with_engine(engine, _PARITY_PROBABILITY_PROGRAM)

    p_cat0 = {"a": 0.3, "b": 0.5, "c": 0.2}
    p_cat1 = {"x": 0.4, "y": 0.6}
    for n1, p1 in p_cat0.items():
        for n2, p2 in p_cat1.items():
            key = ("joint", (n1,), (n2,))
            assert pytest.approx(out_by_query[key], abs=1e-5) == p1 * p2


@pytest.mark.skipif(not JanusGrounder.is_available(), reason="janus_swi not available")
def test_janus_within_ad_conjunction_zeroed_by_mvsdd():
    # JanusGrounder emits the formula (leaf_a ∧ leaf_b) for `?- a, b.`
    # because tabling skips per-proof mutex checks. MV-SDD compilation
    # reduces [var=0] ∧ [var=1] to ⊥, so the AC outputs 0 — matching
    # what the k-best prover produces by dropping the conflicting proof
    # at construction time.
    code = """
    0.3::a; 0.7::b.
    impossible :- a, b.
    ?- impossible.
    """
    program = tuple(str_to_rules(code))
    factory = CircuitFactory()
    result = Solver(JanusGrounder()).get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    mod = circuit_to_module(
        *transform_expectation_to_probability(
            *nodes,
            leaf_mapping=build_leaf_mapping(result.labels),
            variables=result.variables,
        ),
        names=answers,
    )
    out = mod(torch.zeros((1, 0)))
    assert pytest.approx(float(out[0, 0]), abs=1e-5) == 0.0


@pytest.mark.parametrize("grounder_cls", [SimpleGrounder, JanusGrounder])
def test_compile_to_module_enforces_the_ad_mutex(grounder_cls):
    """The ordinary ``compile_to_module`` path also compiles the AD mutex in.

    Neither ``SimpleGrounder`` nor ``JanusGrounder`` drops conflicting AD
    branches during search, so ``compile_to_module`` must carry
    ``EngineResult.variables`` through the probability transform and into the
    compile backend. Treating the branches as independent leaves instead gives
    ``P(a) * P(b)`` for two branches of one annotated disjunction.
    """
    if grounder_cls is JanusGrounder and not JanusGrounder.is_available():
        pytest.skip("janus_swi not available")
    code = """
    0.3::a; 0.7::b.
    impossible :- a, b.
    either :- a.
    either :- b.
    ?- impossible.
    ?- either.
    """
    program = tuple(str_to_rules(code))
    result = Solver(grounder_cls()).get_query_result(program, CircuitFactory())
    module = compile_to_module(result, DeepLogModuleFactory())

    # Constant labels are baked in, so the module takes no runtime input.
    assert list(module.get_input_shape()) == []
    # Query answers are labelled with the algebra of the values they name, so
    # ``to_dict`` is keyed by the labelled atom.
    values = to_dict(module(), module.get_output_shape())
    impossible = with_structure(("impossible",), "probability")
    either = with_structure(("either",), "probability")
    assert pytest.approx(float(values[impossible]), abs=1e-5) == 0.0
    assert pytest.approx(float(values[either]), abs=1e-5) == 1.0


# --- Recognition: the variable is read back off the ground atoms --------------


def _recognize(code: str):
    """Ground ``code`` and return the recognized variables."""
    program = tuple(str_to_rules(code))
    return (
        Solver(SimpleGrounder()).get_query_result(program, CircuitFactory()).variables
    )


def test_recognized_variable_has_a_domain_of_values_not_atoms():
    """Branches sharing an atom template are one variable over the argument values.

    ``digit(i1,0); ...; digit(i1,2)`` is the paper's ``digit(i1,N)`` with ``N``
    over the digits, so the *variable* is ``digit(i1)``, its *domain* is the
    values ``0,1,2`` — not the atoms asserting them — and it *occurs* in
    ``digit(i1,_)``. No value is paired with an atom: the atom asserting a value
    is what substituting it yields.
    """
    variables = _recognize(
        """
        0.2::digit(i1,0); 0.3::digit(i1,1); 0.5::digit(i1,2).
        ?- digit(i1,1).
        """
    )

    (variable,) = variables
    assert variable.name == ("digit", ("i1",))
    assert variable.domain.values == (("0",), ("1",), ("2",))
    assert variables[variable] == (("digit", ("i1",), ("_",)),)
    occurrence = variables[variable][0]
    assert atom_asserting(occurrence, ("1",)) == ("digit", ("i1",), ("1",))


def test_one_variable_per_instance_of_a_disjunction():
    """Two images are two variables: they bind the branches differently."""
    variables = _recognize(
        """
        0.4::digit(i1,0); 0.6::digit(i1,1).
        0.7::digit(i2,0); 0.3::digit(i2,1).
        ?- digit(i1,0).
        ?- digit(i2,0).
        """
    )

    assert {variable.name for variable in variables} == {
        ("digit", ("i1",)),
        ("digit", ("i2",)),
    }


def test_unrelated_branches_are_still_one_variable():
    """``0.3::a; 0.7::b.`` has no argument to read a value from, so it is positional.

    Recognition still yields *one* variable — that is what the exclusivity rests
    on — but the whole atom is its term position, so the occurrence is bare and
    the values *are* the branch atoms. Its name is minted, because unrelated
    predicates give the variable itself no user-side referent.
    """
    variables = _recognize("0.3::a; 0.7::b.\n?- a.")

    (variable,) = variables
    assert variable.name[0] == "@variable"
    assert variable.domain.values == (("a",), ("b",))
    assert variables[variable] == (OPEN,)
    assert atom_asserting(OPEN, ("b",)) == ("b",)


def test_a_program_without_a_disjunction_recognizes_nothing():
    """Recognition is optional: a plain program declares no variable at all.

    Its leaves are then two-valued variables of their own, which is exactly what
    the plain SDD compiler already gives them.
    """
    assert _recognize("0.3::a.\n0.7::b.\n?- a.") == {}


def test_the_declaration_does_not_depend_on_what_was_reached():
    """A branch the grounder never proved changes nothing about the declaration.

    The disjunction declares all three values whether or not a proof reaches
    them, and every one of them has an atom — reachability is the *circuit's*
    fact, which is why nothing here records it. The unreached values become the
    residual the compiler reads as the complement.
    """
    variables = _recognize(
        """
        0.2::digit(i1,0); 0.3::digit(i1,1); 0.5::digit(i1,2).
        ?- digit(i1,1).
        """
    )

    (variable,) = variables
    assert len(variable) == 3
    assert set(indicated_values(variables)) == {
        ("digit", ("i1",), ("0",)),
        ("digit", ("i1",), ("1",)),
        ("digit", ("i1",), ("2",)),
    }


def _neural_factory():
    """A factory resolving ``m_digit(image, value)`` to the image's own column."""

    class _Passthrough(torch.nn.Module):
        def forward(self, x):
            return x

    return DeepLogModuleFactory(
        atom_builders={
            ("m_digit", 2, "probability"): get_network_predicate(
                "m_digit", 2, "probability", _Passthrough()
            )
        }
    )


_SHARED_DIGIT = """
both(X,Y) :- digit(X,N), digit(Y,N).
?- both(i1,i2).
"""
_NEURAL = "nn(m_digit, [X], Y, [0..2]) :: digit(X,Y)." + _SHARED_DIGIT
_ENUMERATED = (
    "m_digit(X,0)::digit(X,0); m_digit(X,1)::digit(X,1); m_digit(X,2)::digit(X,2)."
    + _SHARED_DIGIT
)


def test_a_neural_annotation_declares_the_variable_it_names():
    """``nn(m_digit,[X],Y,[0..2])`` says outright what the ground form implies.

    ``Y`` is the variable and ``[0..2]`` its domain, so nothing has to be read
    back off the branches: the declaration is ``digit(X,_)`` over ``0,1,2``, and
    grounding gives one variable per image.
    """
    variables = _recognize(_NEURAL)

    assert {variable.name for variable in variables} == {
        ("digit", ("i1",)),
        ("digit", ("i2",)),
    }
    for variable, occurrences in variables.items():
        assert variable.domain.values == (("0",), ("1",), ("2",))
        assert occurrences == ((*variable.name, OPEN),)


def test_a_neural_annotation_is_the_enumerated_disjunction():
    """The non-ground spelling is sugar, so both compile to the same number.

    ``nn(m_digit,[X],Y,[0..2]) :: digit(X,Y).`` *is*
    ``m_digit(X,0)::digit(X,0); ... ; m_digit(X,2)::digit(X,2).`` — same
    variables, same labels, same probability.
    """
    assert _recognize(_NEURAL).keys() == _recognize(_ENUMERATED).keys()

    def probability(code):
        program = tuple(str_to_rules(code))
        result = Solver(SimpleGrounder()).get_query_result(program, CircuitFactory())
        module = compile_to_module(result, _neural_factory())
        table = {("i1",): [0.2, 0.3, 0.5], ("i2",): [0.6, 0.1, 0.3]}
        # One batch row, one symbol (the image), whose value is its class scores.
        arguments = [
            torch.tensor([[table[next(iter(get_all_symbols(shape)))]]])
            for shape in as_tuple(module.get_input_shape())
        ]
        return to_dict(module(*arguments), module.get_output_shape())

    # 0.2*0.6 + 0.3*0.1 + 0.5*0.3 — exclusivity is what keeps the cross terms out.
    assert probability(_NEURAL) == probability(_ENUMERATED)
    (value,) = probability(_NEURAL).values()
    assert value == pytest.approx(0.3)


def test_a_neural_annotation_must_name_a_variable_of_its_atom():
    """The declared variable has to occur in the atom, or it opens no position."""
    with pytest.raises(ValueError, match="must occur exactly once"):
        _recognize("nn(m_digit, [X], Y, [0..2]) :: digit(X,Z).\n?- digit(i1,0).")

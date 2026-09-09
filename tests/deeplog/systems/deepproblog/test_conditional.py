#  Copyright (c) 2024-2026. KU Leuven
"""End-to-end tests for conditional queries P(q | e) in DeepProbLog.

Evidence is expressed with the integrity-constraint operator: ``:- body.``
conditions the distribution on ``¬body`` (the dual of the ``?- q.`` query
directive). So positive evidence ``e`` is written ``:- not(e).`` and negative
evidence ``:- e.``. The engine builds, per answer, the joint ``q∧e`` and the
shared evidence ``e``; :func:`compile_to_module` compiles all of them into one
module and divides with a :class:`~deeplog.module.ColumnwiseModule` to obtain
``P(q | e) = E[q∧e] / E[e]``.

Numeric-constant labels (``0.6::burglary``) are baked into the compiled AC by
the knowledge-compilation backend, so the alarm programs here have no runtime
inputs at all — ``_evaluate`` feeds only the empty ``(batch, 0)`` channel and
the declared probabilities below are informational, not supplied at forward time.
"""

import torch

import deeplog.circuit.knowledge_compile.sdd as sdd
from deeplog.formula import DeepLogModuleFactory
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import str_to_rules
from deeplog.shape import SymTensor
from deeplog.shape import get_all_symbols
from deeplog.symbol import without_structure
from deeplog.systems.deepproblog import Solver
from deeplog.systems.deepproblog import compile_to_module
from deeplog.util import as_tuple


# The textbook alarm network: alarm fires on a burglary or an earthquake.
ALARM = """
0.6::burglary.
0.3::earthquake.
alarm :- burglary.
alarm :- earthquake.
"""
ALARM_PROBS = {"burglary": 0.6, "earthquake": 0.3}


def _conditional_module(program_text: str):
    """Build the P(q | e) module for ``program_text`` with the SimpleGrounder.

    Conditioning is a branch of the ordinary query path: ``get_query_result``
    conditions on any ``:- body.`` constraints and ``compile_to_module`` divides
    by the evidence when it is present.
    """
    program = tuple(str_to_rules(program_text))
    result = Solver(SimpleGrounder()).get_query_result(program, CircuitFactory())
    return result, compile_to_module(result, DeepLogModuleFactory())


def _evaluate(module, values: dict[str, float]) -> dict[str, float]:
    """Run ``module`` with per-atom probabilities, returning answer -> value.

    Inputs are supplied by symbol name; the composed module may expose them as a
    single ``SymTensor`` or (for multiple queries) a tuple of one tensor per
    symbol, so one tensor is built per input ``SymTensor``.
    """
    args = [
        torch.tensor([[values[sym[1][0]] for sym in get_all_symbols(symtensor)]])
        for symtensor in as_tuple(module.get_input_shape())
    ]
    output = module(*args) if args else module()
    flat = torch.cat([tensor.flatten() for tensor in as_tuple(output)])
    # Outputs are labelled probabilities, so the answer atom is read through the
    # tag — the same way the inputs are read above.
    names = [
        without_structure(sym)[0] for sym in get_all_symbols(module.get_output_shape())
    ]
    return dict(zip(names, flat.tolist(), strict=True))


def test_positive_evidence_alarm():
    """P(burglary | alarm) = P(b) / P(alarm) = 0.6 / (1 - 0.4*0.7) = 0.8333…."""
    _, module = _conditional_module(ALARM + ":- not(alarm).\n?- burglary.")
    result = _evaluate(module, ALARM_PROBS)
    torch.testing.assert_close(result["burglary"], 0.6 / (1 - 0.4 * 0.7))


def test_negative_evidence_dependent():
    """P(burglary | ¬alarm) = 0: alarm follows from burglary, so they cannot coexist."""
    _, module = _conditional_module(ALARM + ":- alarm.\n?- burglary.")
    result = _evaluate(module, ALARM_PROBS)
    torch.testing.assert_close(result["burglary"], 0.0)


def test_negative_evidence_independent():
    """P(burglary | ¬earthquake) = P(burglary) = 0.6 (the two facts are independent)."""
    _, module = _conditional_module(ALARM + ":- earthquake.\n?- burglary.")
    result = _evaluate(module, ALARM_PROBS)
    torch.testing.assert_close(result["burglary"], 0.6)


def test_evidence_makes_query_certain():
    """P(alarm | burglary) = 1.0: a burglary deterministically triggers the alarm."""
    _, module = _conditional_module(ALARM + ":- not(burglary).\n?- alarm.")
    result = _evaluate(module, ALARM_PROBS)
    torch.testing.assert_close(result["alarm"], 1.0)


def test_no_constraints_is_unconditional_pq():
    """With no constraints, evidence is None and the result is the plain P(q)."""
    result, module = _conditional_module(ALARM + "?- alarm.")
    assert result.evidence is None
    # P(alarm) = 1 - (1 - 0.6)(1 - 0.3) = 0.72, the unconditional success probability.
    torch.testing.assert_close(_evaluate(module, ALARM_PROBS)["alarm"], 0.72)


def test_impossible_evidence_stays_finite():
    """Contradictory evidence (P(e)=0) yields a finite value via the denominator clamp."""
    # `:- a.` conditions on ¬a; `:- not(a).` conditions on a; together impossible.
    _, module = _conditional_module("0.5::a.\n:- a.\n:- not(a).\n?- a.")
    value = _evaluate(module, {"a": 0.5})["a"]
    assert torch.isfinite(torch.tensor(value))


def test_multiple_queries_one_constraint():
    """Two queries share one evidence denominator; each gets its own posterior."""
    _, module = _conditional_module(
        ALARM + ":- not(alarm).\n?- burglary.\n?- earthquake."
    )
    assert len(list(get_all_symbols(module.get_output_shape()))) == 2
    result = _evaluate(module, ALARM_PROBS)
    # P(b | alarm) = 0.6/0.72; P(e | alarm) = 0.3/0.72.
    torch.testing.assert_close(result["burglary"], 0.6 / 0.72)
    torch.testing.assert_close(result["earthquake"], 0.3 / 0.72)


def test_conjunctive_constraint_body():
    """`:- a, b.` conditions on ¬(a∧b): P(a | ¬(a∧b)) = (Pa - Pa*Pb)/(1 - Pa*Pb)."""
    _, module = _conditional_module("0.5::a.\n0.5::b.\n:- a, b.\n?- a.")
    result = _evaluate(module, {"a": 0.5, "b": 0.5})
    torch.testing.assert_close(result["a"], (0.5 - 0.25) / (1 - 0.25))


def test_conditional_and_unconditional_agree_on_output_shape():
    """Adding a constraint to a program must not reshape its result.

    Both paths declare one ``SymTensor`` with a column per query answer, so a
    consumer that reads the result by name — or by shape — keeps working when a
    ``:- body.`` constraint is added to an otherwise unchanged program.
    """
    queries = "?- burglary.\n?- earthquake."
    _, unconditional = _conditional_module(ALARM + queries)
    _, conditional = _conditional_module(ALARM + ":- not(alarm).\n" + queries)

    assert isinstance(unconditional.get_output_shape(), SymTensor)
    assert conditional.get_output_shape() == unconditional.get_output_shape()


def test_conditional_compiles_the_answers_and_evidence_together(monkeypatch):
    """N answers plus the evidence are *one* knowledge compilation, not N+1.

    Lowering each numerator separately would throw away the co-residence the
    batched WMC transform just achieved: every answer would get its own copy of
    each predicate module (running a neural predicate once per answer) and would
    recompute the shared evidence count.
    """
    roots_per_call = []
    original = sdd.compile_sdd

    def counting_compile_sdd(circuit, roots, *args, **kwargs):
        roots_per_call.append(len(roots))
        return original(circuit, roots, *args, **kwargs)

    monkeypatch.setattr(sdd, "compile_sdd", counting_compile_sdd)
    _conditional_module(ALARM + ":- not(alarm).\n?- burglary.\n?- earthquake.")

    # One call, holding both numerators q∧e and the shared denominator e.
    assert roots_per_call == [3]


def test_fully_baked_posterior_is_callable_bare():
    """Constant labels leave no runtime input, so `module()` must still work.

    The posterior is the outermost module over the compiled circuit, and shape
    validation runs there first — so it, not just the wrapper beneath it, has to
    synthesise the empty batch channel.
    """
    _, module = _conditional_module(ALARM + ":- not(alarm).\n?- burglary.")
    assert list(get_all_symbols(module.get_input_shape())) == []
    torch.testing.assert_close(module().flatten().tolist()[0], 0.6 / 0.72)


def test_no_reserved_output_name_is_introduced():
    """The evidence column is told apart by its own name, not a reserved atom.

    The conditional path used to relabel the denominator to a synthetic
    ``("@evidence",)`` symbol so the divide could pick it out. Column selection
    is by symbol, so the denominator simply keeps the positional name the
    lowering gave it — nothing ``@``-prefixed is minted into the atom space.
    """
    _, module = _conditional_module(ALARM + ":- not(alarm).\n?- burglary.")

    names = [str(symbol) for symbol in get_all_symbols(module.get_output_shape())]
    assert not any(name.startswith("@") for name in names)


def test_shared_circuit_is_evaluated_once_for_all_answers():
    """One forward evaluates the joint module once, not once per answer.

    This is what dividing *inside* a single module buys: N separately-lowered
    numerators over a separately-lowered denominator would re-run the shared
    circuit — and the predicate modules feeding it — for every answer.
    """
    _, module = _conditional_module(
        ALARM + ":- not(alarm).\n?- burglary.\n?- earthquake.\n?- alarm."
    )

    calls = []
    inner = module._inner
    original_forward = inner.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    inner.forward = counting_forward
    module()

    assert sum(calls) == 1

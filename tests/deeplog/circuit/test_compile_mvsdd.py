#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the MV-SDD compile path (categorical drop-in for PySDD)."""

import pytest
import torch

from deeplog.circuit import to_module
from deeplog.formula import DeepLogModuleFactory
from deeplog.systems.deepproblog.engine import SimpleEngine
from deeplog.systems.deepproblog.program import str_to_rules


pymvsdd = pytest.importorskip("pymvsdd")


def _compile_program(code, structure_override="probability"):
    """Compile an AD program through MV-SDD.

    Defaults to the probability semiring (real, with AND=product /
    OR=sum) — that's the only semantics that makes sense for the
    probabilistic facts these tests are exercising. The default
    boolean (Gödel) semiring is left as an opt-in for tests that
    specifically want it.
    """
    program = tuple(str_to_rules(code))
    factory = DeepLogModuleFactory()
    result = SimpleEngine().get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    return (
        to_module(
            *(n.root for n in nodes),
            names=answers,
            categoricals=result.categoricals,
            labels=result.labels,
            structure_override=structure_override,
        ),
        result,
    )


def test_constant_ad_bakes_branches_into_circuit():
    # Every AD branch's label is a numeric constant — the compiler
    # bakes those values into the AC, so the resulting torch module
    # takes NO runtime inputs. The output is exactly each branch's
    # declared probability.
    mod, _ = _compile_program(
        """
        0.3::a; 0.5::b; 0.2::c.
        ?- a.
        ?- b.
        ?- c.
        """
    )
    # No input slots — declared constants 0.3/0.5/0.2 live inside
    # the AC.
    assert list(mod.get_input_shape()) == []
    assert list(mod.get_output_shape()) == [("a",), ("b",), ("c",)]
    # WrappedModule with vmap=True needs at least a batch axis; pass
    # an empty trailing dim.
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.3, 0.5, 0.2]]), atol=1e-5)


def test_constant_ad_in_rule_body_outputs_marginal():
    # Rule body composes an AD branch into a derived predicate. With
    # both `heads` and `tails` carrying numeric labels, no runtime
    # input is needed; `?- win.`'s marginal is just P(heads) = 0.4.
    mod, _ = _compile_program(
        """
        0.4::heads; 0.6::tails.
        win :- heads.
        ?- win.
        """
    )
    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.4]]), atol=1e-5)


def test_unresolved_labels_remain_input_slots():
    # The compiler bakes a label into the AC iff it's a numeric
    # constant. Non-numeric labels (e.g. ``net(0)``) have no value
    # at compile time, so the AC keeps them as free leaves whose
    # truth indicator the caller fills at evaluation time. In real
    # DeepProbLog use this is the slot a neural network's per-class
    # softmax flows into — exercised end-to-end by the MNIST tests
    # below; here we just pin down the input-slot contract.
    mod, _ = _compile_program(
        """
        net(0)::a; net(1)::b; net(2)::c.
        ?- a.
        ?- b.
        ?- c.
        """
    )
    assert list(mod.get_input_shape()) == [
        ("@cat", ("@cat_id_0",), ("0",)),
        ("@cat", ("@cat_id_0",), ("1",)),
        ("@cat", ("@cat_id_0",), ("2",)),
    ]
    # The AC isn't computable without runtime inputs — there's no
    # numeric value associated with `net(0)`/`net(1)`/`net(2)`, so the
    # caller has to feed truth indicators (or neural softmax outputs)
    # for each (cat_id, value) slot.
    out = mod(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0]]))
    out = mod(torch.tensor([[0.3, 0.5, 0.2]]))
    assert torch.allclose(out, torch.tensor([[0.3, 0.5, 0.2]]), atol=1e-5)


def test_independent_constant_categoricals_conjoin():
    # Two independent constant-labeled ADs; query their conjunction.
    # With both ADs baked in and the probability semiring (AND=product,
    # OR=sum), the output is the joint probability of the two chosen
    # branches: P(a) · P(c).
    mod, _ = _compile_program(
        """
        0.3::a; 0.7::b.
        0.4::c; 0.6::d.
        win :- a, c.
        ?- win.
        """
    )
    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.3 * 0.4]]), atol=1e-5)


def _build_mnist_addition_program():
    """The classic n=2 MNIST addition program, with per-image ADs."""

    def ad(img):
        # `classifier(img, 0) :: digit(img, 0); ...; classifier(img, 9) :: digit(img, 9).`
        return (
            "; ".join(f"classifier({img},{n}) :: digit({img},{n})" for n in range(10))
            + "."
        )

    return (
        ad("i1")
        + "\n"
        + ad("i2")
        + "\n"
        + "addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), "
        "digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).\n" + "?- addition(i1,i2,S).\n"
    )


def _addition_inputs(mod, p1, p2):
    """Build the 20-slot input tensor matching mod's declared input order."""
    inputs = [0.0] * len(list(mod.get_input_shape()))
    for i, sym in enumerate(mod.get_input_shape()):
        cat_id = sym[1][0]
        value = int(sym[2][0])
        # cat_id_0 is i1's classifier, cat_id_1 is i2's (allocation order
        # is alphabetic on cat_id strings, but the underlying engine
        # assigns them in fact-emission order — `i1` first, then `i2`).
        inputs[i] = (p1 if cat_id == "@cat_id_0" else p2)[value]
    return torch.tensor([inputs])


def test_mnist_addition_n2_one_hot():
    # n=2 MNIST addition. With one-hot inputs at digits i1=3, i2=5, the
    # AC should output 1.0 only at addition(i1,i2,8) and 0.0 elsewhere.
    code = _build_mnist_addition_program()
    program = tuple(str_to_rules(code))
    factory = DeepLogModuleFactory()
    result = SimpleEngine().get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)
    # 19 possible sums (0..18).
    assert len(answers) == 19

    mod = to_module(
        *(n.root for n in nodes),
        names=answers,
        categoricals=result.categoricals,
        structure_override="probability",
    )

    i1_gt, i2_gt = 3, 5
    p1 = [1.0 if d == i1_gt else 0.0 for d in range(10)]
    p2 = [1.0 if d == i2_gt else 0.0 for d in range(10)]
    out = mod(_addition_inputs(mod, p1, p2))
    out_by_sum = {
        int(name[3][0]): float(v)
        for name, v in zip(answers, out[0].tolist(), strict=True)
    }
    for s in range(19):
        expected = 1.0 if s == i1_gt + i2_gt else 0.0
        assert pytest.approx(out_by_sum[s], abs=1e-5) == expected


def test_mnist_addition_n2_soft_distribution():
    # Same program, with peaked-but-not-saturated softmax inputs. The
    # addition's marginal must equal the convolution
    # P(sum=s) = Σ_{n1+n2=s} P_classifier_1(n1) · P_classifier_2(n2).
    code = _build_mnist_addition_program()
    program = tuple(str_to_rules(code))
    factory = DeepLogModuleFactory()
    result = SimpleEngine().get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)

    mod = to_module(
        *(n.root for n in nodes),
        names=answers,
        categoricals=result.categoricals,
        structure_override="probability",
    )

    # Two simulated softmax outputs.
    p1 = [0.9 if d == 3 else 0.1 / 9 for d in range(10)]
    p2 = [0.9 if d == 5 else 0.1 / 9 for d in range(10)]
    out = mod(_addition_inputs(mod, p1, p2))
    out_by_sum = {
        int(name[3][0]): float(v)
        for name, v in zip(answers, out[0].tolist(), strict=True)
    }

    # Expected: classic discrete convolution.
    for s in range(19):
        expected = sum(p1[n1] * p2[s - n1] for n1 in range(10) if 0 <= s - n1 <= 9)
        assert pytest.approx(out_by_sum[s], abs=1e-5) == expected
    # And the marginal sums to (sum p1) * (sum p2) = 1.
    assert pytest.approx(sum(out_by_sum.values()), abs=1e-5) == 1.0


def test_categoricals_threaded_through_to_module():
    # Sanity check that to_module's `categoricals` arg actually routes
    # to the MV-SDD path — without it, the input shape would be 1 slot
    # per leaf (klay default), not 1 per (var, value).
    program = tuple(
        str_to_rules(
            """
            0.3::a; 0.7::b.
            ?- a.
            """
        )
    )
    factory = DeepLogModuleFactory()
    result = SimpleEngine().get_query_result(program, factory)
    answers, nodes = zip(*result.formulas.items(), strict=True)

    mod_cats = to_module(
        *(n.root for n in nodes), names=answers, categoricals=result.categoricals
    )
    mod_no_cats = to_module(*(n.root for n in nodes), names=answers)

    # MV-SDD path: one slot per cat value (here the AD's b is unused so
    # only "a" is in the circuit; padded to 2 slots).
    mvsdd_inputs = list(mod_cats.get_input_shape())
    assert all(sym[0] == "@cat" for sym in mvsdd_inputs)

    # Default (klay) path: one slot per leaf.
    klay_inputs = list(mod_no_cats.get_input_shape())
    assert len(klay_inputs) == 1

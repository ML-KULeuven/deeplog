#  Copyright (c) 2024-2026. KU Leuven
"""Tests for deeplog.algebraic — structures, operators, and the registry."""

import math

import pytest
import torch

from deeplog.algebraic import BOOLEAN
from deeplog.algebraic import LOGPROBABILITY
from deeplog.algebraic import PROBABILITY
from deeplog.algebraic import Algebra
from deeplog.algebraic import AlgebraicStructure
from deeplog.algebraic import Semifield
from deeplog.algebraic import Semiring
from deeplog.algebraic import get_algebraic_structure
from deeplog.algebraic import register_structure
from deeplog.algebraic import structure_registry
from deeplog.circuit import Circuit
from deeplog.circuit.backends import register_klay_semiring
from deeplog.circuit.backends import routed_operators
from deeplog.circuit.backends import select_backend
from deeplog.circuit.knowledge_compile import knowledge_compile
from deeplog.symbol import FalseSymbol
from deeplog.symbol import TrueSymbol
from deeplog.variable import Domain


# --- AlgebraicStructure ---


def test_algebraic_structure_operators():
    s = AlgebraicStructure(
        name="test",
        operator_fns={"op1": lambda a, b: a + b, "op2": lambda a: -a},
    )
    assert s.operators == frozenset({"op1", "op2"})


def test_algebraic_structure_get_operator_fn():
    def fn(a, b):
        return a * b

    s = AlgebraicStructure(name="test", operator_fns={"mul": fn})
    assert s.get_operator_fn("mul") is fn
    assert s.get_operator_fn("missing") is None


def test_default_constant_fn_parses_numbers():
    s = AlgebraicStructure(name="test")
    assert s.get_constant_value(("3.14",)) == pytest.approx(3.14)
    assert s.get_constant_value(("0",)) == 0.0
    assert s.get_constant_value(("abc",)) is None
    assert s.get_constant_value(("a", "b")) is None


def test_custom_constant_fn():
    s = AlgebraicStructure(
        name="test",
        constant_fn=lambda sym: 42.0 if sym == ("magic",) else None,
    )
    assert s.get_constant_value(("magic",)) == 42.0
    assert s.get_constant_value(("other",)) is None


# --- Semiring ---


def test_semiring_has_product_and_sum():
    sr = Semiring(name="test_sr", product="mul", sum="add")
    assert sr.product == "mul"
    assert sr.sum == "add"
    assert "mul" in sr.operators
    assert "add" in sr.operators


def test_semiring_zero_and_one():
    sr = Semiring(name="test_sr", product="mul", zero=("-inf",), one=("0",))
    assert sr.zero == ("-inf",)
    assert sr.one == ("0",)


def test_a_named_constant_resolves_to_the_element_it_names():
    """A named constant's symbol is the spelling of a value, through the same
    ``constant_fn`` every other constant is read with.

    A structure that breaks this says nothing about what its identity *is*,
    which is what an evaluator bakes in.
    """
    with pytest.raises(ValueError, match="spells its 'zero' as"):
        Semiring(name="unresolvable", zero=("z",))


def test_semiring_default_operators():
    sr = Semiring(name="test_sr", product="mul", sum="add")
    a = torch.tensor(2.0)
    b = torch.tensor(3.0)
    mul_fn = sr.get_operator_fn("mul")
    add_fn = sr.get_operator_fn("add")
    assert mul_fn(a, b) == pytest.approx(6.0)
    assert add_fn(a, b) == pytest.approx(5.0)


# --- Algebra ---


def test_algebra_has_negation():
    alg = Algebra(name="test_alg", product="mul", negation="neg")
    assert alg.negation == "neg"
    assert "neg" in alg.operators


def test_algebra_negation_fn():
    alg = Algebra(name="test_alg", product="mul", negation="neg")
    neg_fn = alg.get_operator_fn("neg")
    assert neg_fn(torch.tensor(0.3)) == pytest.approx(0.7)


# --- Built-in structures ---


class TestBoolean:
    def test_name(self):
        assert BOOLEAN.name == "boolean"

    def test_and(self):
        fn = BOOLEAN.get_operator_fn("and")
        assert fn(torch.tensor(1.0), torch.tensor(0.0)) == 0.0
        assert fn(torch.tensor(1.0), torch.tensor(1.0)) == 1.0

    def test_or(self):
        fn = BOOLEAN.get_operator_fn("or")
        assert fn(torch.tensor(0.0), torch.tensor(0.0)) == 0.0
        assert fn(torch.tensor(1.0), torch.tensor(0.0)) == 1.0

    def test_not(self):
        fn = BOOLEAN.get_operator_fn("not")
        assert fn(torch.tensor(1.0)) == 0.0
        assert fn(torch.tensor(0.0)) == 1.0

    def test_roles(self):
        assert BOOLEAN.product == "and"
        assert BOOLEAN.sum == "or"
        assert BOOLEAN.negation == "not"


class TestProbability:
    def test_name(self):
        assert PROBABILITY.name == "probability"

    def test_times(self):
        fn = PROBABILITY.get_operator_fn("times")
        assert fn(torch.tensor(0.5), torch.tensor(0.4)) == pytest.approx(0.2)

    def test_plus(self):
        fn = PROBABILITY.get_operator_fn("plus")
        assert fn(torch.tensor(0.3), torch.tensor(0.2)) == pytest.approx(0.5)

    def test_negate(self):
        fn = PROBABILITY.get_operator_fn("negate")
        assert fn(torch.tensor(0.7)) == pytest.approx(0.3)


class TestLogProbability:
    def test_name(self):
        assert LOGPROBABILITY.name == "logprobability"

    def test_times_is_addition(self):
        fn = LOGPROBABILITY.get_operator_fn("times")
        a, b = torch.tensor(-1.0), torch.tensor(-2.0)
        assert fn(a, b) == pytest.approx(-3.0)

    def test_plus_is_logaddexp(self):
        fn = LOGPROBABILITY.get_operator_fn("plus")
        a, b = torch.tensor(-1.0), torch.tensor(-2.0)
        expected = torch.logaddexp(a, b)
        assert fn(a, b) == pytest.approx(expected.item())

    def test_negate(self):
        fn = LOGPROBABILITY.get_operator_fn("negate")
        p = 0.3
        log_p = math.log(p)
        result = fn(torch.tensor(log_p))
        assert result == pytest.approx(math.log(1 - p), abs=1e-6)


# --- Registry ---


def test_get_algebraic_structure_builtin():
    assert get_algebraic_structure("boolean") is BOOLEAN
    assert get_algebraic_structure("probability") is PROBABILITY
    assert get_algebraic_structure("logprobability") is LOGPROBABILITY


def test_get_algebraic_structure_unknown():
    with pytest.raises(ValueError, match="Unknown structure 'nonexistent'"):
        get_algebraic_structure("nonexistent")


def test_register_and_lookup(monkeypatch):
    # Use monkeypatch to restore structure_registry after the test
    original = dict(structure_registry)
    monkeypatch.setattr("deeplog.algebraic.structure_registry", dict(original))

    custom = AlgebraicStructure(name="custom_test")
    register_structure(custom)
    assert get_algebraic_structure("custom_test") is custom


def test_register_overwrites(monkeypatch):
    original = dict(structure_registry)
    monkeypatch.setattr("deeplog.algebraic.structure_registry", dict(original))

    first = AlgebraicStructure(name="overwrite_test")
    second = AlgebraicStructure(name="overwrite_test")
    register_structure(first)
    register_structure(second)
    assert get_algebraic_structure("overwrite_test") is second


# --- Circuit operators and the semifield ------------------------------------


def test_the_generic_evaluator_routes_every_operator():
    """Nothing narrows a structure the generic evaluator will compile.

    ``lower_generic`` evaluates a circuit node-by-node straight out of
    ``operator_fns``, so every operator the structure defines is a node it can
    walk and nothing is ever cut out of it.
    """
    fuzzy = AlgebraicStructure(
        name="fuzzy_ops",
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "implies": lambda a, b: 1 - a + a * b,
        },
    )

    assert select_backend(Circuit(fuzzy)) == "generic"
    assert routed_operators(fuzzy, "generic") == fuzzy.operators


def test_the_backend_walks_route_the_semiring_roles():
    """Klay and knowledge compilation build one node per role, by role not by name.

    ``and``/``or``/``not`` and ``times``/``plus``/``negate`` are the same three
    nodes under two spellings, which is why neither backend needs a table of
    operator names.
    """
    for backend in ("klay", "kc"):
        assert routed_operators(BOOLEAN, backend) == frozenset({"and", "or", "not"})
        assert routed_operators(PROBABILITY, backend) == frozenset(
            {"times", "plus", "negate"}
        )


def test_a_division_is_a_circuit_node_that_no_backend_walk_routes():
    """The algebra defines it, the circuit carries it, the compiler cuts it.

    Nothing between the two is allowed to know that ``divide`` is special: a
    semifield declares a quotient without knowing what compiles it, and the
    circuit accepts the node like any other.
    """
    assert "divide" in PROBABILITY.operators
    assert "divide" not in routed_operators(PROBABILITY, "klay")
    assert "divide" not in routed_operators(PROBABILITY, "kc")

    circuit = Circuit("probability")
    node = circuit.get_operator("divide")(
        circuit.get_leaf_node(("a",)), circuit.get_leaf_node(("b",))
    )
    assert circuit._get_node(node).node_type == "divide"


def test_the_backend_never_depends_on_the_graph():
    """Growing the graph moves no circuit onto a different backend.

    A division and a numeric constant are exactly the two things that used to:
    the first by narrowing what the circuit was allowed to carry, the second by
    demoting the whole circuit to the evaluator that could carry it.
    """
    circuit = Circuit("probability")
    a = circuit.get_leaf_node(("a",))

    assert select_backend(circuit) == "klay"

    circuit.get_operator("divide")(a, circuit.get_leaf_node(("b",)))
    circuit.get_leaf_node(("_", ("0.5",), ("probability",)))

    assert circuit.constant_values
    assert select_backend(circuit) == "klay"


def test_a_circuit_is_evaluated_as_written_whatever_produced_it():
    """Backend selection reads the algebra and nothing else.

    A weighted model count is not a *reading* a circuit carries — it is what
    knowledge compilation plus a transform produce, after which the result is an
    ordinary arithmetic circuit evaluated as written like any other.
    """
    boolean = Circuit("boolean")
    a, b = boolean.get_leaf_node(("a",)), boolean.get_leaf_node(("b",))
    compiled, node_map = knowledge_compile(boolean, [boolean.get_operator("or")(a, b)])
    counted, _ = compiled.transform(list(node_map.values()), "probability")

    assert select_backend(Circuit("probability")) == "klay"
    assert select_backend(compiled) == "klay"
    assert select_backend(counted) == "klay"


def test_knowledge_compilation_needs_the_logical_roles():
    """It builds a diagram out of the product, sum and complement.

    A structure that declares none of them has no logical reading to compile, so
    it is refused where the diagram would be built rather than deep inside one.
    """
    fuzzy = AlgebraicStructure(name="fuzzy_kc", operator_fns={"op": lambda a, b: a * b})
    circuit = Circuit(fuzzy)
    a, b = circuit.get_leaf_node(("a",)), circuit.get_leaf_node(("b",))
    root = circuit.get_operator("op")(a, b)

    with pytest.raises(ValueError, match="no diagram node"):
        knowledge_compile(circuit, [root])


def test_klay_cannot_implement_a_structure_with_no_product_and_sum():
    """Registration is the claim that a Klay semiring *is* this structure.

    A structure declaring neither a product nor a sum has nothing for a semiring
    to implement, and Klay builds its circuit out of exactly those roles — so the
    claim is refused where it is made rather than failing inside a backend walk.
    """
    fuzzy = AlgebraicStructure(
        name="fuzzy_unregisterable", operator_fns={"and": lambda a, b: a * b}
    )

    with pytest.raises(ValueError, match="declares no product and sum"):
        register_klay_semiring(fuzzy, "real")


def test_semifield_division_divides():
    """The probability semifield's divide is ordinary division."""
    divide = PROBABILITY.get_operator_fn("divide")
    torch.testing.assert_close(
        divide(torch.tensor([0.4]), torch.tensor([0.5])), torch.tensor([0.8])
    )


def test_semifield_division_clamps_a_zero_denominator():
    """Impossible evidence stays finite rather than becoming NaN/inf."""
    divide = PROBABILITY.get_operator_fn("divide")
    assert torch.isfinite(divide(torch.tensor([0.0]), torch.tensor([0.0]))).all()


def test_log_semifield_division_agrees_with_probability():
    """Log-space division is subtraction, transported along ``log``."""
    log_divide = LOGPROBABILITY.get_operator_fn("divide")
    result = log_divide(torch.tensor([0.4]).log(), torch.tensor([0.5]).log())
    torch.testing.assert_close(result.exp(), torch.tensor([0.8]))


def test_the_division_floor_scales_with_the_tensor_dtype():
    """Regression: ``1e-12`` is exactly ``0.0`` in float16, so the clamp did nothing."""
    assert torch.tensor(1e-12, dtype=torch.float16).item() == 0.0

    divide = PROBABILITY.get_operator_fn("divide")
    for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
        one = torch.ones(1, dtype=dtype)
        zero = torch.zeros(1, dtype=dtype)
        assert torch.isfinite(divide(one, zero)).all(), dtype
        assert not torch.isnan(divide(zero, zero)).any(), dtype


def test_log_operators_are_the_log_image_of_the_probability_ones():
    """Every log-space operator must equal ``log`` of its linear counterpart.

    Over 300 decades: spot checks at ``0`` and ``-inf`` cannot see a floor that
    bites in the middle of the range.
    """
    exponents = torch.linspace(-300, 0, 400, dtype=torch.float64)
    p = (10.0**exponents).clamp(max=1.0)
    q = (10.0 ** exponents.flip(0)).clamp(max=1.0)

    for linear, log_space in (
        (PROBABILITY.product_fn, LOGPROBABILITY.product_fn),
        (PROBABILITY.sum_fn, LOGPROBABILITY.sum_fn),
        (PROBABILITY.division_fn, LOGPROBABILITY.division_fn),
    ):
        reference = linear(p, q).log()
        finite = torch.isfinite(reference)
        torch.testing.assert_close(
            log_space(p.log(), q.log())[finite], reference[finite]
        )

    # The complement is defined on [0, 1] and loses all precision at the ends.
    within = torch.linspace(1e-8, 1 - 1e-8, 400, dtype=torch.float64)
    torch.testing.assert_close(
        LOGPROBABILITY.negation_fn(within.log()),
        PROBABILITY.negation_fn(within).log(),
    )


def test_log_division_does_not_clamp_ordinary_log_probabilities():
    """Regression: flooring at ``log(tiny)`` (``-87.3`` in float32) made
    ``P(1e-50 | 1e-40)`` come back ~118x too small.
    """
    log_divide = LOGPROBABILITY.get_operator_fn("divide")
    joint = torch.tensor([math.log(1e-50)])
    evidence = torch.tensor([math.log(1e-40)])

    torch.testing.assert_close(
        log_divide(joint, evidence), joint - evidence, rtol=0, atol=0
    )
    assert math.isclose(
        math.exp(log_divide(joint, evidence).item()), 1e-10, rel_tol=1e-4
    )


def test_the_log_division_floor_scales_with_the_tensor_dtype():
    """The transported floor keeps ``-inf - -inf`` from going NaN in every dtype."""
    log_divide = LOGPROBABILITY.get_operator_fn("divide")
    for dtype in (torch.float64, torch.float32, torch.float16):
        neg_inf = torch.full((1,), float("-inf"), dtype=dtype)
        assert not torch.isnan(log_divide(neg_inf, neg_inf)).any(), dtype


def test_log_semifield_division_clamps_a_zero_denominator():
    """The denominator floor transports through the log isomorphism.

    In log space the invariant is *not NaN* rather than finite: ``-inf`` is the
    legitimate representation of probability zero. Without the clamp, dividing
    an impossible numerator by impossible evidence is ``-inf - -inf = nan``;
    the clamp floors the denominator at ``log(eps)`` so the result is ``-inf``,
    i.e. zero.
    """
    log_divide = LOGPROBABILITY.get_operator_fn("divide")
    zero = torch.tensor([0.0]).log()

    assert not torch.isnan(log_divide(zero, zero)).any()
    assert torch.isnan(zero - zero).all()  # what the clamp is protecting against


def test_a_circuit_rejects_an_operator_its_algebra_does_not_define():
    """The algebra is the whole of it: an operator it never declared is not a node."""
    circuit = Circuit("probability")
    with pytest.raises(ValueError, match="not an operator of"):
        circuit.get_operator("implies")


def test_probability_circuits_stay_on_the_klay_fast_path():
    """Regression: a semifield's extra operator must not disqualify Klay.

    Backend selection asks whether Klay implements the *structure*, not whether
    every operator the structure defines happens to be Klay-routable. Testing the
    full operator set would make ``divide`` disqualify every probability circuit
    and silently drop it onto ``lower_generic`` — no error, just slower.
    """
    assert select_backend(Circuit("probability")) == "klay"
    assert select_backend(Circuit("boolean")) == "klay"


def test_probability_is_both_an_algebra_and_a_semifield():
    """The two extensions are independent, so a structure can declare both."""
    assert isinstance(PROBABILITY, Algebra)
    assert isinstance(PROBABILITY, Semifield)
    assert isinstance(LOGPROBABILITY, Algebra)
    assert isinstance(LOGPROBABILITY, Semifield)


def test_boolean_is_an_algebra_but_not_a_semifield():
    """min/max has no multiplicative inverse, so BOOLEAN is not a semifield.

    The old ``Semifield(Algebra)`` chain could not express this: division and a
    complement were welded together.
    """
    assert isinstance(BOOLEAN, Algebra)
    assert not isinstance(BOOLEAN, Semifield)


def test_boolean_declares_its_two_values():
    """A structure is a set of values plus operators, so BOOLEAN knows its two.

    Definition 1 writes ``B = ({true, false}, {not}, {or, and})``, where the
    first component is the set of labels a formula takes. A variable associated
    with a structure inherits those as its domain, which is what makes "binary"
    derived rather than hardcoded.
    """
    assert BOOLEAN.values == (FalseSymbol, TrueSymbol)
    assert Domain.of_structure(BOOLEAN).values == (FalseSymbol, TrueSymbol)


def test_a_number_is_not_a_boolean_value():
    """Boolean's constants are the two values it declares, not numerals.

    An atom still resolves to ``None``: not a constant is a different answer
    from an ill-formed one.
    """
    with pytest.raises(ValueError, match="not a value of 'boolean'"):
        BOOLEAN.get_constant_value(("0",))
    assert BOOLEAN.get_constant_value(("a",)) is None


def test_a_structure_with_unenumerable_values_declares_none():
    """Probability ranges over an interval, so it hands out no domain at all.

    Returning a plausible domain here would let a variable over ``probability``
    be silently enumerated as if it were boolean.
    """
    assert PROBABILITY.values is None
    assert LOGPROBABILITY.values is None
    with pytest.raises(ValueError, match="declares no values"):
        Domain.of_structure(PROBABILITY)


def test_a_structure_klay_does_not_implement_is_not_evaluated_in_real():
    """Regression: the semiring lookup must raise, not fall back to ``real``.

    A custom structure whose operators happen to be *spelled* like Klay's would
    otherwise be handed to Klay's real semiring with its own ``operator_fns``
    discarded — no error, wrong numbers.
    """
    from deeplog.circuit.backends import klay_semiring

    custom = Algebra(name="not_registered_with_klay", product="and", sum="or")
    with pytest.raises(ValueError, match="No Klay semiring is registered"):
        klay_semiring(custom)

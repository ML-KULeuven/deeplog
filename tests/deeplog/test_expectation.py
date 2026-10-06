#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the expectation aggregation.

``expectation(X; P): φ`` is the expectation of ``φ`` over ``X`` distributed as
``P``: a factor weighs each named value of its binder, and the mass they leave
is a missing value's, where every leaf over the binder is false. Without ``P``,
the binders are independent, and each value ``φ`` tests a binder for weighs as
that test in probability. The default builder
counts (:func:`~deeplog.formula.lowering.expectation.weighted_model_count`) and
enumerates what it cannot count
(:func:`~deeplog.formula.lowering.expectation.enumeration`), and
:func:`~deeplog.formula.lowering.expectation.sampling` returns one that
estimates it.
"""

import math
import sys
from functools import partial

import pytest
import torch

from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Compiler
from deeplog import Domain
from deeplog import EqualityPredicate
from deeplog import NetworkPredicate
from deeplog import Predicate
from deeplog import Transformation
from deeplog import UnaryOp
from deeplog import enumeration
from deeplog import parse_formula
from deeplog import reshape
from deeplog import sampling
from deeplog import score_function
from deeplog import with_structure
from deeplog.formula.ast import fold
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.module.aggregation_modules import ExpectationModule
from deeplog.shape import SymTensor
from deeplog.shape import get_all_symbols

from .testing_modules import Forecast


FALSE = ("false",)
TRUE = ("true",)

BURGLARY = ("Burglary",)
EARTHQUAKE = ("Earthquake",)

BURGLARY_ATOM = ("=", BURGLARY, TRUE)
EARTHQUAKE_ATOM = ("=", EARTHQUAKE, TRUE)

BURGLARY_BOOL_SYM = ("_", BURGLARY_ATOM, ("boolean",))
EARTHQUAKE_BOOL_SYM = ("_", EARTHQUAKE_ATOM, ("boolean",))

BURGLARY_PROB_SYM = ("_", BURGLARY_ATOM, ("probability",))
EARTHQUAKE_PROB_SYM = ("_", EARTHQUAKE_ATOM, ("probability",))


def _expectation(binders, child, params=()):
    """Build an ``expectation`` AST node."""
    return Aggregation("expectation", tuple(binders), tuple(params), child)


@pytest.fixture
def compilations(monkeypatch):
    """The number of roots of each knowledge compilation, in order."""
    import deeplog.circuit.knowledge_compilation.mvsdd as mvsdd
    import deeplog.circuit.knowledge_compilation.sdd as sdd

    roots_per_call = []

    def counting(original):
        def compile_counted(circuit, roots, *args, **kwargs):
            roots_per_call.append(len(roots))
            return original(circuit, roots, *args, **kwargs)

        return compile_counted

    monkeypatch.setattr(sdd, "compile_sdd", counting(sdd.compile_sdd))
    monkeypatch.setattr(mvsdd, "compile_mvsdd", counting(mvsdd.compile_mvsdd))
    return roots_per_call


# --- Without a distribution, a binder weighs the values its body tests ------


def test_expectation_boolean_disjunction():
    """Expectation of Burglary OR Earthquake compiles to probability semiring."""
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    exp = _expectation([BURGLARY, EARTHQUAKE], disjunction)

    module = reshape(
        Compiler().compile(exp),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # Inputs are probability values for each atom
    # With P(B=true)=0.5, P(E=true)=0.5 (uniform):
    # E[B or E] = 0.75
    result = module(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.75]]))

    # With P(B=true)=0.8, P(E=true)=0.3:
    # E[B or E] = 1 - P(B=false)*P(E=false) = 1 - 0.2*0.7 = 0.86
    result2 = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(result2, torch.tensor([[0.86]]))


def test_expectation_single_variable():
    """Expectation over a single boolean variable."""
    # A trivial circuit by double negation (equivalent to the bare atom).
    b_circuit = UnaryOp("not", UnaryOp("not", Atom(BURGLARY_BOOL_SYM)))
    exp = _expectation([BURGLARY], b_circuit)

    module = reshape(Compiler().compile(exp), input=SymTensor([BURGLARY_PROB_SYM]))

    assert module.get_input_shape() == SymTensor([BURGLARY_PROB_SYM])
    assert len(list(module.get_output_shape())) == 1

    # E[B] with P(B=true)=0.5 -> 0.5
    result = module(torch.tensor([[0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.5]]))

    # E[B] with P(B=true)=0.7 -> 0.7
    result2 = module(torch.tensor([[0.7]]))
    torch.testing.assert_close(result2, torch.tensor([[0.7]]))


def test_expectation_conjunction():
    """Expectation of Burglary AND Earthquake."""
    conjunction = BinaryOp("and", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    exp = _expectation([BURGLARY, EARTHQUAKE], conjunction)

    module = reshape(
        Compiler().compile(exp),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # E[B and E] with uniform (0.5, 0.5) = 0.25
    result = module(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.25]]))


def _builders(builder: str) -> Compiler:
    if builder == "counted":
        return Compiler()
    return Compiler(aggregation_builders={"expectation": enumeration})


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
@pytest.mark.parametrize(("operator", "expected"), [("or", 0.9), ("and", 0.0)])
def test_two_values_of_a_binder_are_exclusive_without_a_distribution(
    builder, operator, expected
):
    """``=(B,true)`` and ``=(B,false)`` each weigh by their own input, as one variable.

    The mass the two leave is ``B``'s missing value, where both are false.
    """
    formula = parse_formula(
        f"expectation(B): =(B,true)_boolean {operator} =(B,false)_boolean"
    )
    true, false = (
        ("_", ("=", ("B",), (value,)), ("probability",)) for value in ("true", "false")
    )

    module = reshape(
        _builders(builder).compile(formula), input=SymTensor([true, false])
    )

    torch.testing.assert_close(
        module(torch.tensor([[0.6, 0.3]])), torch.tensor([[expected]])
    )


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_a_leaf_over_no_binder_is_not_random_without_a_distribution(builder):
    """A leaf no binder reaches is evaluated as written, as under a distribution."""
    formula = parse_formula("expectation(B): =(B,true)_boolean and a_boolean")

    module = reshape(
        _builders(builder).compile(formula),
        input=SymTensor(["=(B,true) _ probability", "a _ boolean"]),
    )

    torch.testing.assert_close(
        module(torch.tensor([[0.6, 1.0], [0.6, 0.0]])), torch.tensor([[0.6], [0.0]])
    )


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_both_spellings_of_a_test_are_one_input_without_a_distribution(builder):
    """``=(X,a)`` and ``=(a,X)`` test ``X`` for one value, weighed by one input."""
    aggregation_builders = {} if builder == "counted" else {"expectation": enumeration}
    compiler = Compiler(
        variables={("X",): Domain.of(["a", "b"])},
        aggregation_builders=aggregation_builders,
    )

    module = compiler.compile(
        parse_formula("expectation(X): =(X,a)_boolean or =(a,X)_boolean")
    )

    assert set(get_all_symbols(module.get_input_shape())) == {
        ("_", ("=", ("X",), ("a",)), ("probability",))
    }
    torch.testing.assert_close(module(torch.tensor([[0.3]])), torch.tensor([[0.3]]))


def test_a_binder_its_body_does_not_test_weighs_nothing_without_a_distribution():
    """A binder the body never tests has no named value, so its mass is all missing."""
    module = Compiler().compile(parse_formula("expectation(B, E): =(B,true)_boolean"))

    assert module.get_input_shape() == (SymTensor(["=(B,true) _ probability"]),)
    torch.testing.assert_close(module(torch.tensor([[0.6]])), torch.tensor([[0.6]]))


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_a_binder_read_but_not_tested_is_refused_without_a_distribution(builder):
    """Only a test ``=(V, v)`` names a value an input can weigh."""
    formula = parse_formula("expectation(X): q(X)_boolean")

    with pytest.raises(ValueError, match="give it a distribution"):
        _builders(builder).compile(formula)


def test_an_expectation_without_a_distribution_is_counted(compilations):
    pytest.importorskip("pymvsdd")
    formula = parse_formula(
        "expectation(B, E): =(B,true)_boolean or =(B,false)_boolean "
        "or =(E,true)_boolean"
    )

    Compiler().compile(formula)

    assert compilations == [1]


def test_registering_enumeration_gives_the_default_numbers_without_compiling(
    compilations,
):
    """Counting is a fast path: enumerating instead changes no number."""
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    exp = _expectation([BURGLARY, EARTHQUAKE], disjunction)
    compiler = Compiler(aggregation_builders={"expectation": enumeration})

    module = reshape(
        compiler.compile(exp),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.3]])), torch.tensor([[0.86]])
    )
    assert compilations == []


# --- With a distribution --------------------------------------------------------


_BURGLARY_OR_EARTHQUAKE = "=(B,true)_boolean or =(E,true)_boolean"
_P = "p(B,0.8)_probability times p(E,0.3)_probability"


def test_a_factor_over_its_binders_domain_is_counted(compilations):
    """``forecast`` reads ``Rain``'s truth values, so counting writes them into it."""
    compiler = Compiler(atom_builders={("forecast", 1, "probability"): Forecast})
    module = compiler.compile(
        parse_formula(
            "expectation(Rain; forecast(Rain)_probability): =(Rain,true)_boolean"
        )
    )

    assert compilations == [1]
    assert list(get_all_symbols(module.get_input_shape())) == []
    torch.testing.assert_close(module(), torch.tensor([[0.2]]))


def test_an_expectation_under_p_factors_is_counted(compilations):
    """``p`` factors, one per binder, are counted in one compilation, with no inputs."""
    formula = parse_formula(f"expectation(B, E; {_P}): {_BURGLARY_OR_EARTHQUAKE}")

    module = Compiler().compile(formula)

    assert compilations == [1]
    assert list(get_all_symbols(module.get_input_shape())) == []
    torch.testing.assert_close(module(), torch.tensor([[1 - 0.2 * 0.7]]))


def test_under_factors_that_sum_to_one_an_expectation_is_its_sum():
    """No mass is missing, so the expectation is ``sum(X): (φ)_probability times P``."""
    expectation = parse_formula(f"expectation(B, E; {_P}): {_BURGLARY_OR_EARTHQUAKE}")
    summed = parse_formula(
        f"sum(B, E): ({_BURGLARY_OR_EARTHQUAKE})_probability times ({_P})"
    )

    torch.testing.assert_close(
        Compiler().compile(expectation)(), Compiler().compile(summed)()
    )


@pytest.mark.parametrize(
    "aggregation_builders",
    [{}, {"expectation": enumeration}],
    ids=["counted", "enumerated"],
)
def test_a_factor_mentioning_no_binder_scales_the_expectation(aggregation_builders):
    formula = parse_formula(
        "expectation(B; p(B,0.8)_probability times q_probability): =(B,true)_boolean"
    )

    module = Compiler(aggregation_builders=aggregation_builders).compile(formula)

    assert list(get_all_symbols(module.get_input_shape())) == [
        with_structure(("q",), "probability")
    ]
    torch.testing.assert_close(module(torch.tensor([[0.5]])), torch.tensor([[0.4]]))


_ABC = Domain.of(["a", "b", "c"])


class _Weight(Predicate):
    """``w(X)``: 0.2, 0.5 and 0.3 for the values ``a``, ``b`` and ``c`` of ``X``."""

    def __init__(self, atoms, domain=_ABC):
        super().__init__(atoms, (domain,))

    def forward_predicate(self, x):
        return torch.tensor([0.2, 0.5, 0.3])[x.long()]


def _categorical_compiler(**aggregation_builders):
    """A compiler declaring ``X`` over ``a, b, c``, weighed by ``w(X)``."""
    return Compiler(
        variables={("X",): Domain.of(["a", "b", "c"])},
        atom_builders={("w", 1, "probability"): _Weight},
        aggregation_builders=aggregation_builders,
    )


class _SubWeight(_Weight):
    """``w(X)``: 0.2, 0.5 and 0.1 for ``a``, ``b`` and ``c``, which leave 0.2 missing."""

    def forward_predicate(self, x):
        return torch.tensor([0.2, 0.5, 0.1])[x.long()]


class _IsB(Predicate):
    """``isb(X)``: whether ``X`` is ``b``, over ``a, b, c``."""

    def __init__(self, atoms):
        super().__init__(atoms, (_ABC,))

    def forward_predicate(self, x):
        return (x == 1).to(torch.get_default_dtype())


def _missing_mass_compiler(builder: str, builders=None):
    """``X`` over ``a, b, c``, weighed by ``w``, and ``Y`` over ``a, b``, by ``v``.

    ``v`` weighs ``a`` and ``b`` as ``w`` does.
    """
    aggregation_builders = {
        "counted": {},
        "enumerated": {"expectation": enumeration},
        "sampled": {"expectation": sampling(40_000)},
    }[builder]
    return Compiler(
        variables={("X",): Domain.of(["a", "b", "c"]), ("Y",): Domain.of(["a", "b"])},
        atom_builders={
            ("w", 1, "probability"): _SubWeight,
            ("v", 1, "probability"): partial(_SubWeight, domain=Domain.of(["a", "b"])),
            **(builders or {}),
        },
        aggregation_builders=aggregation_builders,
    )


@pytest.mark.parametrize("builder", ["counted", "enumerated", "sampled"])
@pytest.mark.parametrize(
    ("body", "value"),
    [
        # b and c, and the missing 0.2, where no test holds.
        ("not =(X,a)_boolean", 0.8),
        ("=(X,a)_boolean or not =(X,b)_boolean", 0.5),
        ("not =(X,b)_boolean and not =(X,c)_boolean", 0.4),
        ("=(X,a)_boolean or =(X,b)_boolean", 0.7),
    ],
)
def test_the_mass_named_values_leave_is_a_missing_value(builder, body, value):
    """A factor summing below one leaves its binder a value no test names."""
    pytest.importorskip("pymvsdd")
    torch.manual_seed(0)
    compiler = _missing_mass_compiler(builder)
    formula = parse_formula(f"expectation(X; w(X)_probability): {body}")

    torch.testing.assert_close(
        compiler.compile(formula)(), torch.tensor([[value]]), atol=0.01, rtol=0
    )


@pytest.mark.parametrize(
    "aggregation_builders",
    [{}, {"expectation": enumeration}, {"expectation": sampling(1)}],
    ids=["counted", "enumerated", "sampled"],
)
def test_a_distribution_without_a_complement_leaves_no_missing_value(
    aggregation_builders,
):
    """In ``real`` the mass ``w`` leaves goes nowhere: b and c weigh 0.5 and 0.1.

    Sampling draws only from a distribution in ``probability``, so it enumerates
    this one, exactly, even from a single draw.
    """
    compiler = Compiler(
        variables={("X",): Domain.of(["a", "b", "c"])},
        atom_builders={("w", 1, "real"): _SubWeight},
        aggregation_builders=aggregation_builders,
    )
    formula = parse_formula("expectation(X; w(X)_real): not =(X,a)_boolean")

    torch.testing.assert_close(compiler.compile(formula)(), torch.tensor([[0.6]]))


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
@pytest.mark.parametrize(
    ("body", "value"),
    [
        # w(a), times the most Y weighs: v(b) or its missing value, 0.5.
        ("=(X,a)_boolean", 0.1),
        # v(b), times the most X weighs: w(b) or its missing value, 0.5.
        ("=(X,a)_boolean or =(Y,b)_boolean", 0.25),
    ],
)
def test_an_expectation_in_mpe_is_its_most_probable_model(builder, body, value):
    """The most a model weighs, its missing values included (glab #160).

    ``w`` and ``v`` weigh as in :func:`_missing_mass_compiler`, under ``max``.
    """
    compiler = Compiler(
        variables={("X",): Domain.of(["a", "b", "c"]), ("Y",): Domain.of(["a", "b"])},
        atom_builders={
            ("w", 1, "mpe"): _SubWeight,
            ("v", 1, "mpe"): partial(_SubWeight, domain=Domain.of(["a", "b"])),
        },
        aggregation_builders={
            "counted": {},
            "enumerated": {"expectation": enumeration},
        }[builder],
    )
    formula = parse_formula(f"expectation(X, Y; w(X)_mpe times v(Y)_mpe): {body}")

    torch.testing.assert_close(compiler.compile(formula)(), torch.tensor([[value]]))


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_a_binder_the_body_never_tests_weighs_one(builder):
    """Its named values and its missing value together weigh one."""
    compiler = _missing_mass_compiler(builder)
    formula = parse_formula(
        "expectation(X, Y; w(X)_probability times v(Y)_probability): =(X,a)_boolean"
    )

    torch.testing.assert_close(compiler.compile(formula)(), torch.tensor([[0.2]]))


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_each_binder_leaves_its_own_missing_mass(builder, compilations):
    """``w`` leaves 0.2 of ``X``'s mass missing, and ``v`` 0.3 of ``Y``'s."""
    pytest.importorskip("pymvsdd")
    formula = parse_formula(
        "expectation(X, Y; w(X)_probability times v(Y)_probability):"
        " not =(X,a)_boolean and not =(X,b)_boolean"
        " and not =(Y,a)_boolean and not =(Y,b)_boolean"
    )

    # X is c or missing, 0.1 + 0.2, and Y is missing, 0.3.
    torch.testing.assert_close(
        _missing_mass_compiler(builder).compile(formula)(), torch.tensor([[0.09]])
    )
    assert compilations == ([1] if builder == "counted" else [])


@pytest.mark.parametrize("n", [4, 7, 8])
def test_binders_whose_factors_differ_only_in_the_binder_weigh_apart(n):
    """``p(Vi,0.5)`` gives every ``Vi`` the same weights, each its own (glab #147)."""
    binders = [f"V{i}" for i in range(n)]
    factors = " times ".join(f"p({binder},0.5)_probability" for binder in binders)
    body = " or ".join(f"=({binder},true)_boolean" for binder in binders)
    formula = parse_formula(f"expectation({', '.join(binders)}; {factors}): {body}")

    torch.testing.assert_close(
        Compiler().compile(formula)(), torch.tensor([[1 - 0.5**n]])
    )


class _Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, -1)


class _LogSoftmax(torch.nn.Module):
    def forward(self, x):
        return torch.log_softmax(x, -1)


def _digit_compiler(algebra: str, builder: str):
    """``X`` over ``a, b, c``, weighed by a softmax network in ``algebra``."""
    network = _LogSoftmax() if algebra == "logprobability" else _Softmax()
    domain = Domain.of(["a", "b", "c"])
    digit = partial(NetworkPredicate, module=network, domain=domain)
    aggregation_builders = {"enumerated": {"expectation": enumeration}}.get(builder, {})
    return Compiler(
        variables={("X",): domain},
        atom_builders={("digit", 2, algebra): digit},
        aggregation_builders=aggregation_builders,
    )


# These logits' softmax sums past one by rounding, in either algebra's sum.
_ROUNDED_PAST_ONE = [[[1.9, -1.4, -0.5]]]


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
@pytest.mark.parametrize(
    ("algebra", "zero"), [("probability", 0.0), ("logprobability", -math.inf)]
)
def test_a_complete_distribution_leaves_no_missing_mass(builder, algebra, zero):
    """Only the missing outcome satisfies the body, and it weighs the algebra's zero."""
    pytest.importorskip("pymvsdd")
    formula = parse_formula(
        f"expectation(X; digit(img,X)_{algebra}):"
        " not =(X,a)_boolean and not =(X,b)_boolean and not =(X,c)_boolean"
    )
    logits = torch.tensor(_ROUNDED_PAST_ONE, requires_grad=True)

    value = _digit_compiler(algebra, builder).compile(formula)(logits)
    value.sum().backward()

    assert value.item() == zero
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_enumeration_counts_as_counting_does_in_log_space():
    """The same value and gradient, where enumeration used to have no cast (glab #149)."""
    formula = parse_formula(
        "expectation(X; digit(img,X)_logprobability): not =(X,a)_boolean"
    )
    results = []
    for builder in ("counted", "enumerated"):
        logits = torch.tensor([[[0.5, -1.0, 2.0]]], requires_grad=True)
        value = _digit_compiler("logprobability", builder).compile(formula)(logits)
        value.sum().backward()
        results.append((value, logits.grad))

    (counted, counted_grad), (enumerated, enumerated_grad) = results
    torch.testing.assert_close(enumerated, counted)
    torch.testing.assert_close(enumerated_grad, counted_grad)


@pytest.mark.parametrize("builder", ["enumerated", "sampled"])
def test_a_leaf_over_a_missing_binder_is_false(builder):
    """``isb(X)`` holds for ``b`` alone, so ``not isb(X)`` holds for a, c and missing."""
    torch.manual_seed(0)
    compiler = _missing_mass_compiler(builder, {("isb", 1, "boolean"): _IsB})
    formula = parse_formula("expectation(X; w(X)_probability): not isb(X)_boolean")

    torch.testing.assert_close(
        compiler.compile(formula)(), torch.tensor([[0.5]]), atol=0.01, rtol=0
    )


def test_a_sum_has_no_missing_value():
    """A sum adds over the named values alone: it is no expectation."""
    formula = parse_formula(
        "sum(X): (not =(X,a)_boolean)_probability times w(X)_probability"
    )

    torch.testing.assert_close(
        _missing_mass_compiler("counted").compile(formula)(), torch.tensor([[0.6]])
    )


def test_a_binder_without_a_factor_is_refused():
    formula = parse_formula(
        "expectation(X, Y; w(X)_probability): =(X,a)_boolean and =(Y,a)_boolean"
    )

    with pytest.raises(ValueError, match="gives no factor"):
        _missing_mass_compiler("counted").compile(formula)


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_values_of_a_binder_are_exclusive_under_a_distribution(builder):
    """Two values of one binder cannot both hold: they are one categorical variable."""
    if builder == "counted":
        pytest.importorskip("pymvsdd")
        compiler = _categorical_compiler()
    else:
        compiler = _categorical_compiler(expectation=enumeration)
    formula = parse_formula(
        "expectation(X; w(X)_probability): =(X,a)_boolean or =(X,b)_boolean"
    )

    torch.testing.assert_close(compiler.compile(formula)(), torch.tensor([[0.7]]))


def test_without_mv_sdd_two_values_of_a_binder_are_enumerated(
    monkeypatch, compilations
):
    """Counting is a fast path: without its knowledge compiler, the same number."""
    monkeypatch.setitem(sys.modules, "pymvsdd", None)
    formula = parse_formula(
        "expectation(X; w(X)_probability): =(X,a)_boolean or =(X,b)_boolean"
    )

    value = _categorical_compiler().compile(formula)()

    torch.testing.assert_close(value, torch.tensor([[0.7]]))
    assert compilations == []


def test_without_mv_sdd_one_value_per_binder_is_still_counted(
    monkeypatch, compilations
):
    """Only a group reaching two values of one binder needs mv-sdd."""
    monkeypatch.setitem(sys.modules, "pymvsdd", None)
    formula = parse_formula(f"expectation(B, E; {_P}): {_BURGLARY_OR_EARTHQUAKE}")

    value = Compiler().compile(formula)()

    torch.testing.assert_close(value, torch.tensor([[1 - 0.2 * 0.7]]))
    assert compilations == [1]


def test_a_network_factor_is_counted_by_its_rows():
    """A network atom with the binder in its value position is a factor."""
    pytest.importorskip("pymvsdd")

    class Passthrough(torch.nn.Module):
        def forward(self, x):
            return x

    domain = Domain.of(["a", "b", "c"])
    compiler = Compiler(
        variables={("X",): domain},
        atom_builders={
            ("digit", 2, "probability"): partial(
                NetworkPredicate, module=Passthrough(), domain=domain
            )
        },
    )
    formula = parse_formula(
        "expectation(X; digit(img,X)_probability): =(X,a)_boolean or =(X,c)_boolean"
    )

    module = compiler.compile(formula)

    assert module.get_input_shape() == (SymTensor([("img",)]),)
    torch.testing.assert_close(
        module(torch.tensor([[[0.2, 0.5, 0.3]]])), torch.tensor([[0.5]])
    )


@pytest.mark.parametrize(
    "aggregation_builders",
    [{}, {"expectation": enumeration}, {"expectation": sampling(40_000)}],
    ids=["counted", "enumerated", "sampled"],
)
def test_a_joint_factor_no_builder_computes_weighs_each_pair_by_its_input(
    aggregation_builders,
):
    """``weather(Rain, Wet)`` is a label per pair of values, not one for all four."""
    torch.manual_seed(0)
    formula = parse_formula(
        "expectation(Rain, Wet; weather(Rain,Wet)_probability): "
        "not =(Rain,true)_boolean or =(Wet,true)_boolean"
    )
    table = {
        "weather(true,true)": 0.4,
        "weather(true,false)": 0.1,
        "weather(false,true)": 0.2,
        "weather(false,false)": 0.3,
    }
    module = reshape(
        Compiler(aggregation_builders=aggregation_builders).compile(formula),
        input=SymTensor([f"{atom} _ probability" for atom in table]),
    )

    # Only rain without wet breaks the rule.
    torch.testing.assert_close(
        module(torch.tensor([list(table.values())])),
        torch.tensor([[0.9]]),
        atol=0.01,
        rtol=0,
    )


def test_a_counted_body_is_never_lowered_as_a_boolean_module():
    """Counting maps the body's leaves to weights: the module holds no value test."""

    class Spy(EqualityPredicate):
        pass

    def spy(atoms):
        return Spy(atoms, lambda _: Domain.of([FALSE, TRUE]))

    formula = parse_formula(f"expectation(B, E; {_P}): {_BURGLARY_OR_EARTHQUAKE}")

    module = Compiler(atom_builders={("=", 2, "boolean"): spy}).compile(formula)

    assert not any(isinstance(inner, Spy) for inner in module.modules())


def test_expectations_whose_factors_disagree_are_counted_apart(compilations):
    """One binder with two distributions is two variables: two compilations."""
    first = parse_formula("expectation(B; p(B,0.8)_probability): =(B,true)_boolean")
    second = parse_formula(
        "expectation(B; p(B,0.4)_probability): =(B,true)_boolean or =(B,true)_boolean"
    )

    module = Compiler().compile(first, second)

    assert compilations == [1, 1]
    torch.testing.assert_close(module(), torch.tensor([[0.8, 0.4]]))


def test_expectation_too_many_params_raises_error():
    """An expectation's one param is its distribution."""
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    pb = Atom(("_", ("prob", BURGLARY, TRUE), ("probability",)))
    pe = Atom(("_", ("prob", EARTHQUAKE, TRUE), ("probability",)))

    exp = _expectation([BURGLARY], disjunction, params=[pb, pe])

    with pytest.raises(ValueError, match="takes one param"):
        Compiler().compile(exp)


def test_expectation_non_boolean_raises_error():
    """Expectation over non-boolean formula should raise an error."""
    X = ("X",)
    prob_pred = ("p", X, ("_", ("0.3",), ("probability",)))
    prob_atom = Atom(("_", prob_pred, ("probability",)))

    exp = _expectation([X], prob_atom)

    with pytest.raises(ValueError, match="boolean"):
        Compiler().compile(exp)


def test_an_expectation_is_a_value_of_its_distributions_algebra():
    """The distribution names the algebra, and probability is the default."""
    body = Atom(BURGLARY_BOOL_SYM)
    in_log = Atom(("_", ("logp", BURGLARY, ("0.8",)), ("logprobability",)))

    assert _expectation([BURGLARY], body).structure == "probability"
    assert _expectation([BURGLARY], body, (in_log,)).structure == "logprobability"


# --- Construction and batching ----------------------------------------------


def test_an_expectation_is_never_a_leaf_of_a_circuit():
    """Construction keeps every aggregation symbolic: it binds variables."""
    constructed = fold(
        _expectation([BURGLARY], Atom(BURGLARY_BOOL_SYM)), CircuitFactory()
    )

    assert isinstance(constructed, Aggregation)


def test_compiling_two_expectations_shares_one_knowledge_compilation(compilations):
    """Two formulas compiled together count their shared circuit once."""
    b, e = Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM)
    disjunction = _expectation([BURGLARY, EARTHQUAKE], BinaryOp("or", b, e))
    conjunction = _expectation([BURGLARY, EARTHQUAKE], BinaryOp("and", b, e))

    module = reshape(
        Compiler().compile(disjunction, conjunction),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert compilations == [2]

    # E[B or E] = 1 - (1-0.8)(1-0.3) = 0.86; E[B and E] = 0.8*0.3 = 0.24
    out = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(out, torch.tensor([[0.86, 0.24]]))


def test_counts_under_different_roots_share_one_knowledge_compilation(compilations):
    """Counts over one circuit are compiled together wherever they sit in a level.

    One count is a root; the other is an operand of a product with a ``sum``, a
    symbolic node, under a second root. Both are in the top level, so the walk
    takes them in one compilation.
    """
    b, e = Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM)
    disjunction = _expectation([BURGLARY, EARTHQUAKE], BinaryOp("or", b, e))
    conjunction = _expectation([BURGLARY, EARTHQUAKE], BinaryOp("and", b, e))
    one = Aggregation("sum", (BURGLARY,), (), Transformation("probability", b))

    module = reshape(
        Compiler().compile(disjunction, BinaryOp("times", conjunction, one)),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert compilations == [2]

    # E[B or E] = 0.86; E[B and E] = 0.24, times one.
    out = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(out, torch.tensor([[0.86, 0.24]]))


# --- Sampling estimates it from drawn assignments -----------------------------


def _sampling(samples: int) -> Compiler:
    return Compiler(aggregation_builders={"expectation": sampling(samples)})


@pytest.mark.parametrize(
    "text",
    [
        f"expectation(B, E; {_P}): {_BURGLARY_OR_EARTHQUAKE}",
        # Without a distribution, each binder weighs the values its body tests.
        f"expectation(B, E): {_BURGLARY_OR_EARTHQUAKE}",
        # One factor over both binders makes them one variable.
        f"expectation(B, E; p(B,0.8)_probability times p(E,0.3)_probability "
        f"times q(B,E)_probability): {_BURGLARY_OR_EARTHQUAKE}",
    ],
)
def test_sampling_estimates_the_expectation(text):
    torch.manual_seed(0)
    formula = parse_formula(text)
    exact, estimate = Compiler().compile(formula), _sampling(20_000).compile(formula)
    symbols = list(get_all_symbols(exact.get_input_shape()))
    assert list(get_all_symbols(estimate.get_input_shape())) == symbols
    if symbols:
        exact, estimate = (
            reshape(module, input=SymTensor(symbols)) for module in (exact, estimate)
        )
    inputs = [torch.full((1, len(symbols)), 0.6)] if symbols else []

    torch.testing.assert_close(estimate(*inputs), exact(*inputs), atol=0.02, rtol=0)


def test_sampled_expectations_over_agreeing_factors_read_the_same_draws():
    """``P(B | B)`` is one on any draws, and on no two independent ones."""
    torch.manual_seed(0)
    formula = parse_formula(
        f"(expectation(B, E; {_P}): =(B,true)_boolean and =(B,true)_boolean) "
        f"divide (expectation(B, E; {_P}): =(B,true)_boolean)"
    )

    module = _sampling(5).compile(formula)

    assert [float(module()) for _ in range(10)] == [1.0] * 10


def test_expectations_whose_factors_disagree_are_sampled_apart():
    formula = parse_formula(
        "(expectation(B; p(B,0.8)_probability): =(B,true)_boolean) times "
        "(expectation(B; p(B,0.3)_probability): =(B,true)_boolean)"
    )

    module = _sampling(10).compile(formula)

    sampled = [m for m in module.modules() if isinstance(m, ExpectationModule)]
    assert [m.samples for m in sampled] == [10, 10]


def test_a_variable_one_expectation_reads_free_is_not_drawn_for_another():
    """``X`` is free in the second and bound by the first, so it stays an input."""
    first = parse_formula("expectation(X): =(X,true)_boolean")
    second = parse_formula("expectation(Y): (=(Y,true)_boolean and =(X,true)_boolean)")

    module = _sampling(10).compile(first, second)

    assert ("X",) in get_all_symbols(module.get_input_shape())


def test_the_score_function_estimates_the_gradient():
    torch.manual_seed(0)
    formula = parse_formula(
        f"expectation(B, E; p(B,pb)_probability times p(E,pe)_probability): "
        f"{_BURGLARY_OR_EARTHQUAKE}"
    )
    labels = SymTensor(
        [with_structure((name,), "probability") for name in ("pb", "pe")]
    )
    exact = reshape(Compiler().compile(formula), input=labels)
    estimate = reshape(_sampling(4_000).compile(formula), input=labels)
    x = torch.tensor([[0.8, 0.3]], requires_grad=True)

    (expected,) = torch.autograd.grad(exact(x).sum(), x)
    estimated = torch.stack(
        [torch.autograd.grad(estimate(x).sum(), x)[0] for _ in range(20)]
    ).mean(dim=0)

    # d/dpb (1 - (1 - pb)(1 - pe)) = 1 - pe, and d/dpe = 1 - pb
    torch.testing.assert_close(expected, torch.tensor([[0.7, 0.2]]))
    torch.testing.assert_close(estimated, expected, atol=0.02, rtol=0)


def test_the_score_function_keeps_the_values():
    values = torch.rand(2, 3, 4)
    log_probability = torch.rand(2, 3, 4, requires_grad=True)

    surrogate = score_function(values, log_probability)

    torch.testing.assert_close(surrogate, values)


def test_one_draw_estimates_the_gradient_without_a_baseline():
    """With no other draws to average, the score function is the draw's own.

    ``B`` is true with probability 1, so the draw is true and its log-probability
    gradient is ``1 / 1``.
    """
    formula = parse_formula("expectation(B): =(B,true)_boolean")
    module = _sampling(1).compile(formula)
    x = torch.tensor([[1.0]], requires_grad=True)

    value = module(x)
    (gradient,) = torch.autograd.grad(value.sum(), x)

    torch.testing.assert_close(value.detach(), torch.tensor([[1.0]]))
    torch.testing.assert_close(gradient, torch.tensor([[1.0]]))


def test_sampling_draws_at_least_one_assignment():
    with pytest.raises(ValueError, match="at least one assignment"):
        sampling(0)

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
from deeplog.algebraic import Semiring
from deeplog.algebraic import get_algebraic_structure
from deeplog.algebraic import register_structure
from deeplog.algebraic import structure_registry


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
    sr = Semiring(name="test_sr", product="mul", zero=("z",), one=("o",))
    assert sr.zero == ("z",)
    assert sr.one == ("o",)
    assert sr.named_constants == {"zero": ("z",), "one": ("o",)}


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

    def test_idempotent(self):
        assert BOOLEAN.idempotent is True


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

    def test_not_idempotent(self):
        assert PROBABILITY.idempotent is False


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

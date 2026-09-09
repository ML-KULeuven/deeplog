#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the domains DeepLog variables range over."""

import pytest
import torch

from deeplog.algebraic import BOOLEAN
from deeplog.algebraic import REAL
from deeplog.symbol import FalseSymbol
from deeplog.symbol import TrueSymbol
from deeplog.variable import Domain
from deeplog.variable import SymbolicDomain
from deeplog.variable import TensorDomain


def test_a_symbolic_domain_enumerates_its_value_positions():
    """A named value's identity is its position, which is what a body is bound to.

    The atom builders resolve a value symbol to the same position, so the two
    halves of an enumeration agree without either restating the domain.
    """
    domain = Domain.of(range(10))
    assert len(domain) == 10
    assert domain.values[3] == ("3",)
    assert domain.index(("3",)) == 3
    assert torch.equal(domain.as_tensor(), torch.arange(10))


def test_domain_of_accepts_symbols_strings_and_numbers():
    assert Domain.of(["a", ("b",), 3]).values == (("a",), ("b",), ("3",))


def test_a_value_outside_the_domain_raises():
    with pytest.raises(ValueError, match="not a value of this domain"):
        Domain.of(["a", "b"]).index(("c",))


def test_a_boolean_variable_inherits_its_domain_from_its_structure():
    """Def 8: a reification variable inherits its domain from its structure."""
    assert Domain.of_structure(BOOLEAN) == SymbolicDomain((FalseSymbol, TrueSymbol))


def test_a_structure_without_values_hands_out_no_domain():
    """REAL defines no operators and no values; it must not stand in for one."""
    with pytest.raises(ValueError, match="declares no values"):
        Domain.of_structure(REAL)


def test_a_tensor_domain_enumerates_its_rows_and_has_no_value_names():
    """An image domain has values but no names, and must not invent them."""
    domain = Domain.of_tensor(torch.randn(10, 2))
    assert isinstance(domain, TensorDomain)
    assert len(domain) == 10
    assert domain.as_tensor().shape == (10, 2)
    with pytest.raises(ValueError, match="have no names"):
        _ = domain.values

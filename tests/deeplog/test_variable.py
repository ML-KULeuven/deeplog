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
    """REAL's values are not enumerable; it must not stand in for a domain."""
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


@pytest.mark.parametrize(
    ("values", "written"),
    [
        (torch.arange(3), ("0", "1", "2")),
        (torch.tensor([2**53, 2**53 + 1]), ("9007199254740992", "9007199254740993")),
        (torch.tensor([0.1, 2.0]), ("0.1", "2.0")),
        (torch.tensor([0.1, 2.0], dtype=torch.float64), ("0.1", "2.0")),
        (torch.tensor([1e20]), ("1e+20",)),
    ],
    ids=["integers", "integers past float64", "float32", "float64", "large"],
)
def test_a_tensor_domain_of_numbers_writes_each_by_its_numeral(values, written):
    """An integer by its digits, a float by the fewest that read back in its dtype."""
    domain = Domain.of_tensor(values)

    assert domain.values == tuple((numeral,) for numeral in written)
    assert [domain.index((numeral,)) for numeral in written] == list(range(len(values)))


def test_a_tensor_domain_reads_only_its_own_numerals():
    """``2.0`` writes the number 2, but not as a domain of integers writes it."""
    with pytest.raises(ValueError, match="2.0 is not a value of this domain"):
        Domain.of_tensor(torch.arange(3)).index(("2.0",))


def test_a_tensor_domain_holding_a_number_twice_writes_none():
    with pytest.raises(ValueError, match="holds 1 twice"):
        _ = Domain.of_tensor(torch.tensor([1, 1, 2])).values


def test_tensor_domains_of_equal_values_are_equal():
    """Two declarations of one range of numbers declare one domain."""
    assert Domain.of_tensor(torch.arange(3)) == Domain.of_tensor(torch.arange(3))
    assert hash(Domain.of_tensor(torch.arange(3))) == hash(
        Domain.of_tensor(torch.arange(3))
    )
    assert Domain.of_tensor(torch.arange(3)) != Domain.of_tensor(torch.arange(1, 4))
    assert Domain.of_tensor(torch.arange(3)) != Domain.of_tensor(torch.arange(4))
    assert TensorDomain(torch.arange(3)) != Domain.of(range(3))

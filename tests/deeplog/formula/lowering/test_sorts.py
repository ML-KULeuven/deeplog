#  Copyright (c) 2024-2026. KU Leuven
"""Each variable's domain, from the sorts of the arguments it is."""

import pytest
import torch

from deeplog import Compiler
from deeplog import Domain
from deeplog import Predicate
from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog import enumeration
from deeplog import parse_formula

from ...testing_modules import VALUES
from ...testing_modules import Forecast


_ABC = Domain.of(["a", "b", "c"])


class _Weight(Predicate):
    """``w(X)``: 0.2, 0.5 and 0.3 for the values ``a``, ``b`` and ``c``."""

    def __init__(self, atoms):
        super().__init__(atoms, (_ABC,))

    def forward_predicate(self, x):
        return torch.tensor([0.2, 0.5, 0.3])[x.long()]


class _Even(Predicate):
    """``even(N)``: whether the number ``N`` is even, read as it is."""

    def __init__(self, atoms):
        super().__init__(atoms, (VALUES,))

    def forward_predicate(self, n):
        return (n % 2 == 0).to(torch.get_default_dtype())


_BUILDERS = {
    ("w", 1, "probability"): _Weight,
    ("forecast", 1, "probability"): Forecast,
    ("even", 1, "boolean"): _Even,
}


def test_a_variable_ranges_over_the_domain_of_the_arguments_it_is():
    """``X`` is declared nowhere: ``w`` reads it over ``a, b, c``."""
    formula = parse_formula(
        "expectation(X; w(X)_probability): =(X,b)_boolean or =(X,c)_boolean"
    )

    module = Compiler(atom_builders=_BUILDERS).compile(formula)

    torch.testing.assert_close(module(), torch.tensor([[0.8]]))


def test_arguments_giving_a_variable_different_domains_are_refused():
    formula = parse_formula(
        "expectation(X; forecast(X)_probability times w(X)_probability): =(X,a)_boolean"
    )

    with pytest.raises(ValueError, match="X ranges over .* as an argument of"):
        Compiler(atom_builders=_BUILDERS).compile(formula)


def test_a_variable_over_named_values_read_as_a_value_is_refused():
    """``even`` reads a number as it is, which a position among names is not."""
    formula = parse_formula(
        "sum(X): w(X)_probability times (even(X)_boolean)_probability"
    )

    with pytest.raises(ValueError, match="even/1 reads it as a value without a name"):
        Compiler(atom_builders=_BUILDERS).compile(formula)


def test_a_bound_variable_only_read_as_a_value_is_declared():
    """Nothing says which numbers ``N`` ranges over until the compiler does."""
    formula = parse_formula("sum(N): (even(N)_boolean)_real")

    with pytest.raises(ValueError, match="nothing gives it a domain"):
        Compiler(atom_builders=_BUILDERS).compile(formula)
    digits = Compiler(
        variables={("N",): Domain.of_tensor(torch.arange(10))},
        atom_builders=_BUILDERS,
    )
    assert digits.compile(formula)().item() == 5


def test_a_variable_only_tested_ranges_over_the_truth_values():
    """As a reification variable inherits the values of its algebra."""
    module = Compiler().compile(parse_formula("sum(B): (=(B,true)_boolean)_real"))

    assert module().item() == 1


def test_a_builder_whose_module_declares_no_sorts_is_refused():
    def build(atoms):
        atoms = list(atoms)
        return WrappedModule(
            lambda: torch.full((1, len(atoms)), 0.5), (), SymTensor(atoms)
        )

    compiler = Compiler(atom_builders={("q", 1, "probability"): build})

    with pytest.raises(TypeError, match="declares no sorts"):
        compiler.compile(parse_formula("q(x)_probability"))


class _Vowel(Predicate):
    """``vowel(X)``: 0.1 and 0.9 for ``a`` and ``b``, which part of ``w``'s values are."""

    def __init__(self, atoms):
        super().__init__(atoms, (Domain.of(["a", "b"]),))

    def forward_predicate(self, x):
        return torch.tensor([0.1, 0.9])[x.long()]


@pytest.mark.parametrize(
    "aggregation_builders",
    [{}, {"expectation": enumeration}],
    ids=["counted", "enumerated"],
)
def test_a_variable_over_part_of_its_arguments_domain_reads_its_own_values(
    aggregation_builders,
):
    """``X`` over ``c, a`` reaches ``w`` as positions 2 and 0 of ``a, b, c``."""
    compiler = Compiler(
        variables={("X",): Domain.of(["c", "a"])},
        atom_builders=_BUILDERS,
        aggregation_builders=aggregation_builders,
    )

    assert compiler.compile(parse_formula("sum(X): w(X)_probability"))().item() == (
        pytest.approx(0.3 + 0.2)
    )
    torch.testing.assert_close(
        compiler.compile(
            parse_formula("expectation(X; w(X)_probability): =(X,c)_boolean")
        )(),
        torch.tensor([[0.3]]),
    )


def test_a_variable_ranges_over_the_domain_every_other_argument_includes():
    """``vowel`` reads ``X`` over ``a, b``, which ``w``'s ``a, b, c`` includes."""
    formula = parse_formula("sum(X): w(X)_real times vowel(X)_real")
    builders = {
        ("w", 1, "real"): _Weight,
        ("vowel", 1, "real"): _Vowel,
    }

    module = Compiler(atom_builders=builders).compile(formula)

    assert module().item() == pytest.approx(0.2 * 0.1 + 0.5 * 0.9)


def test_arguments_whose_domains_do_not_nest_are_refused():
    builders = {
        ("vowel", 1, "real"): _Vowel,
        ("forecast", 1, "real"): Forecast,
    }

    with pytest.raises(ValueError, match="neither includes the other"):
        Compiler(atom_builders=builders).compile(
            parse_formula("sum(X): vowel(X)_real times forecast(X)_real")
        )


def test_a_declared_domain_its_argument_does_not_include_is_refused():
    compiler = Compiler(
        variables={("X",): Domain.of(["a", "d"])}, atom_builders=_BUILDERS
    )

    with pytest.raises(ValueError, match="which does not include it"):
        compiler.compile(parse_formula("sum(X): w(X)_probability"))


def test_a_variable_over_part_of_a_tensor_domain_reads_its_values_as_they_are():
    """Unnamed values are themselves, so nothing is translated."""

    class Double(Predicate):
        def __init__(self, atoms):
            super().__init__(atoms, (Domain.of_tensor(torch.arange(10)),))

        def forward_predicate(self, n):
            return (2 * n).to(torch.get_default_dtype())

    compiler = Compiler(
        variables={("N",): Domain.of_tensor(torch.tensor([3, 7]))},
        atom_builders={("double", 1, "real"): Double},
    )

    assert compiler.compile(parse_formula("sum(N): double(N)_real"))().item() == 20


def test_aggregations_binding_one_name_range_over_their_own_arguments():
    """The ``X`` of each sum is the argument it is there: ``a, b, c``, and ``a, b``."""
    compiler = Compiler(
        atom_builders={**_BUILDERS, ("vowel", 1, "probability"): _Vowel}
    )
    weights, vowels = (
        parse_formula(f"sum(X): {predicate}(X)_probability")
        for predicate in ("w", "vowel")
    )
    both = parse_formula(
        "(sum(X): w(X)_probability) plus (sum(X): vowel(X)_probability)"
    )

    torch.testing.assert_close(
        compiler.compile(weights, vowels)(), torch.tensor([[1.0, 1.0]])
    )
    torch.testing.assert_close(compiler.compile(both)(), torch.tensor([[2.0]]))

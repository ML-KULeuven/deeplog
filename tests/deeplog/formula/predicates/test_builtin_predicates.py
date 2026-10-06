#  Copyright (c) 2026. KU Leuven
"""How ``=`` reads a written value against the domain of the variable it is compared with."""

import pytest
import torch

from deeplog import Compiler
from deeplog import Domain
from deeplog import parse_formula


def _compiler() -> Compiler:
    letters = Domain.of(["a", "b", "c"])
    return Compiler(
        variables={
            ("C",): Domain.of(["red", "blue"]),
            ("X",): letters,
            ("Y",): letters,
            ("N",): Domain.of_tensor(torch.arange(5)),
        }
    )


@pytest.mark.parametrize(
    ("text", "value"),
    [
        # A variable over unnamed values is compared with the number written.
        ("sum(N): (=(N,2)_boolean)_real", 1.0),
        # The written value may stand on either side.
        ("sum(C): (=(red,C)_boolean)_real", 1.0),
        # Two variables over one domain agree at each of its values.
        ("sum(X, Y): (=(X,Y)_boolean)_real", 3.0),
        # Two written values compare by name.
        ("(=(a,a)_boolean)_real", 1.0),
        ("(=(a,b)_boolean)_real", 0.0),
    ],
)
def test_equality_reads_each_side_in_the_domain_it_is_compared_with(text, value):
    module = _compiler().compile(parse_formula(text))

    torch.testing.assert_close(module(), torch.tensor([[value]]))


def test_two_variables_over_different_domains_are_refused():
    with pytest.raises(ValueError, match="range over different domains"):
        _compiler().compile(parse_formula("sum(C, X): (=(C,X)_boolean)_real"))

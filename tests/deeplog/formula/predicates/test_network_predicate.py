#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the network predicate: the rows it computes, and the atoms read from them.

The predicate reads only the first arguments of the atoms it is given and
computes every value of each, one row of the module's output per distinct
first argument. An atom is the entry of its first argument's row at its value's
position in the predicate's domain.
"""

import pytest
import torch
from torch import nn

from deeplog import Compiler
from deeplog import Domain
from deeplog import NetworkPredicate
from deeplog import SymTensor
from deeplog import parse_formula
from deeplog import reshape
from deeplog import with_structure


class RecordingClassifier(nn.Module):
    """Deterministic classifier that records the batch size of every call.

    Row ``v`` maps to ``[v * weight, v * weight + 1, ..., v * weight + n - 1]``,
    so each (input, class) pair has an easily predicted value.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.n_classes = n_classes
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.rows_seen: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rows_seen.append(x.shape[0])
        offsets = torch.arange(self.n_classes, dtype=x.dtype, device=x.device)
        return x * self.weight + offsets


_CLASSES = Domain.of(range(3))


def _digit(image, value):
    return with_structure(("digit", (image,), (value,)), "probability")


def _inputs(*batch_values):
    """One input row per distinct image, for each batch item."""
    return torch.tensor([[[v] for v in values] for values in batch_values])


def test_it_computes_every_value_of_each_first_argument():
    """Asked for one value of ``i1`` and a variable of ``i2``, it computes all six."""
    predicate = NetworkPredicate(
        [_digit("i1", "2"), _digit("i2", "D")],
        module=RecordingClassifier(),
        domain=_CLASSES,
    )

    assert predicate.get_input_shape() == (SymTensor([("i1",), ("i2",)]),)
    assert predicate.get_output_shape() == SymTensor(
        [_digit(image, str(v)) for image in ("i1", "i2") for v in range(3)]
    )


def test_its_module_runs_once_on_the_distinct_first_arguments():
    module = RecordingClassifier()
    predicate = NetworkPredicate(
        [_digit("i1", "0"), _digit("i1", "1"), _digit("i2", "0")],
        module=module,
        domain=_CLASSES,
    )

    result = predicate(_inputs([5.0, 7.0]))

    # digit(image, n) = value * weight + n, for value in (5, 7) and n in 0..2.
    assert module.rows_seen == [2]
    torch.testing.assert_close(
        result, torch.tensor([[10.0, 11.0, 12.0, 14.0, 15.0, 16.0]])
    )


def test_it_keeps_the_batch_dimension():
    module = RecordingClassifier()
    predicate = NetworkPredicate(
        [_digit("i1", "0"), _digit("i2", "0")], module=module, domain=_CLASSES
    )

    result = predicate(_inputs([5.0, 7.0], [1.0, 3.0]))

    assert module.rows_seen == [4]  # two batch items, two distinct images each
    expected = torch.tensor(
        [[10.0, 11.0, 12.0, 14.0, 15.0, 16.0], [2.0, 3.0, 4.0, 6.0, 7.0, 8.0]]
    )
    torch.testing.assert_close(result, expected)


def test_gradients_reach_the_module():
    module = RecordingClassifier()
    predicate = NetworkPredicate(
        [_digit("i1", "0"), _digit("i2", "0")], module=module, domain=_CLASSES
    )

    predicate(_inputs([5.0, 7.0])).sum().backward()

    # d/dweight of sum over the six entries of (value * weight + n).
    assert module.weight.grad is not None
    torch.testing.assert_close(module.weight.grad, torch.tensor(3 * (5.0 + 7.0)))


def _compiled(domain, value):
    """``digit(i1, value)`` compiled over ``domain``, ``value`` a variable or written."""
    compiler = Compiler(
        atom_builders={
            ("digit", 2, "probability"): lambda atoms: NetworkPredicate(
                atoms, module=RecordingClassifier(), domain=domain
            )
        }
    )
    return compiler.compile(parse_formula(f"digit(i1,{value})_probability"))


@pytest.mark.parametrize(
    ("domain", "value", "row"),
    [
        (Domain.of(range(1, 4)), "3", 2),
        (Domain.of(["zero", "one", "two"]), "one", 1),
        (Domain.of_tensor(torch.tensor([1, 2, 3])), "3", 2),
        (Domain.of_tensor(torch.tensor([0.5, 1.5, 2.5])), "1.5", 1),
    ],
    ids=["numbers from 1", "names", "tensor of numbers", "tensor of floats"],
)
def test_a_written_value_reads_the_row_of_its_position_in_the_domain(
    domain, value, row
):
    """The scores of an image of 5 are 10, 11 and 12."""
    module = _compiled(domain, value)

    assert module(torch.tensor([[[5.0]]])).item() == 10.0 + row


@pytest.mark.parametrize(
    ("domain", "given", "row"),
    [(Domain.of_tensor(torch.tensor([1, 2, 3])), 3.0, 2), (_CLASSES, 1, 1)],
    ids=["a number", "a named value's position"],
)
def test_a_variable_is_given_a_value_and_reads_the_row_of_its_position(
    domain, given, row
):
    """``N`` is free, so the caller gives its value, and the atom reads that row."""
    module = reshape(
        _compiled(domain, "N"), input=(SymTensor([("i1",)]), SymTensor([("N",)]))
    )

    assert module(torch.tensor([[[5.0]]]), torch.tensor([[given]])).item() == 10.0 + row


@pytest.mark.parametrize(
    ("domain", "value", "message"),
    [
        (Domain.of(range(1, 4)), "0", "0 is not a value of the domain of digit/2"),
        (_CLASSES, "uncle", "uncle is not a value of the domain of digit/2"),
        (
            Domain.of_tensor(torch.tensor([1, 2, 3])),
            "2.0",
            "2.0 is not a value of the domain of digit/2",
        ),
    ],
    ids=["outside the domain", "a name", "another numeral"],
)
def test_a_written_value_that_reads_no_row_is_refused(domain, value, message):
    with pytest.raises(ValueError, match=message):
        NetworkPredicate(
            [_digit("i1", value)], module=RecordingClassifier(), domain=domain
        )


def test_a_given_value_that_reads_no_row_is_refused():
    module = reshape(
        _compiled(Domain.of_tensor(torch.tensor([1, 2, 3])), "N"),
        input=(SymTensor([("i1",)]), SymTensor([("N",)])),
    )

    with pytest.raises(ValueError, match="given 0.0, which is not a value"):
        module(torch.tensor([[[5.0]]]), torch.tensor([[0.0]]))


def test_a_written_number_as_the_input_is_refused():
    with pytest.raises(ValueError, match="the number 5 names none"):
        NetworkPredicate(
            [_digit("5", "0")], module=RecordingClassifier(), domain=_CLASSES
        )


def test_a_module_without_a_row_per_value_is_refused():
    predicate = NetworkPredicate(
        [_digit("i1", "2")], module=RecordingClassifier(), domain=Domain.of(range(4))
    )

    with pytest.raises(ValueError, match="where its domain has 4 values"):
        predicate(_inputs([5.0]))


@pytest.mark.parametrize(
    ("domain", "message"),
    [
        (Domain.of_tensor(torch.tensor([1, 1, 2])), "holds 1 twice"),
        (Domain.of_tensor(torch.randn(3, 2)), "have no names"),
        (Domain.of_values(), "have no names"),
    ],
    ids=["a number twice", "vectors", "the values"],
)
def test_a_domain_that_writes_no_values_is_refused(domain, message):
    with pytest.raises(ValueError, match=message):
        NetworkPredicate(
            [_digit("i1", "1")], module=RecordingClassifier(), domain=domain
        )

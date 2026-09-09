#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the network predicate, in particular its sharing of inputs.

Ground atoms that share a first argument (``digit(i1,0) … digit(i1,2)``) are
distinct evaluations over the same input, so the predicate asks for that image
once and the wrapped module sees one row per distinct argument rather than one
per evaluation.
"""

import torch
from torch import nn

from deeplog import SymTensor
from deeplog import get_network_predicate


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


def _predicate(module, images):
    """The ``digit/2`` predicate over ``images x {0, 1, 2}``."""
    arguments = [
        ((image,), (str(n),)) for image in images for n in range(module.n_classes)
    ]
    return get_network_predicate("digit", 2, "probability", module)(arguments)


def _inputs(*batch_values):
    """One input row per distinct image, for each batch item."""
    images = torch.tensor([[[v] for v in values] for values in batch_values])
    return images, torch.zeros(len(batch_values), 0)


def test_predicate_asks_for_each_image_once():
    predicate = _predicate(RecordingClassifier(), ("i1", "i2"))

    # Six evaluations, two images: the interface names each image once.
    assert predicate.get_input_shape() == (
        SymTensor([("i1",), ("i2",)]),
        SymTensor([]),
    )


def test_module_runs_once_per_distinct_first_argument():
    module = RecordingClassifier()
    predicate = _predicate(module, ("i1", "i2"))

    predicate(*_inputs([5.0, 7.0]))

    assert module.rows_seen == [2]


def test_shared_inputs_give_the_per_evaluation_values():
    module = RecordingClassifier()
    predicate = _predicate(module, ("i1", "i2"))

    result = predicate(*_inputs([5.0, 7.0]))

    # digit(image, n) = value * weight + n, for value in (5, 7) and n in 0..2.
    expected = torch.tensor([[10.0, 11.0, 12.0, 14.0, 15.0, 16.0]])
    torch.testing.assert_close(result, expected)


def test_shared_inputs_preserve_the_batch_dimension():
    module = RecordingClassifier()
    predicate = _predicate(module, ("i1", "i2"))

    result = predicate(*_inputs([5.0, 7.0], [1.0, 3.0]))

    assert module.rows_seen == [4]  # two batch items, two distinct images each
    expected = torch.tensor(
        [[10.0, 11.0, 12.0, 14.0, 15.0, 16.0], [2.0, 3.0, 4.0, 6.0, 7.0, 8.0]]
    )
    torch.testing.assert_close(result, expected)


def test_distinct_first_arguments_are_all_evaluated():
    module = RecordingClassifier(n_classes=1)
    predicate = _predicate(module, ("i1", "i2", "i3"))

    result = predicate(*_inputs([5.0, 7.0, 9.0]))

    assert module.rows_seen == [3]  # nothing shared, nothing to skip
    torch.testing.assert_close(result, torch.tensor([[10.0, 14.0, 18.0]]))


def test_gradients_reach_the_module_through_shared_inputs():
    module = RecordingClassifier()
    predicate = _predicate(module, ("i1", "i2"))

    predicate(*_inputs([5.0, 7.0])).sum().backward()

    # d/dweight of sum over the six evaluations of (value * weight + n).
    assert module.weight.grad is not None
    torch.testing.assert_close(module.weight.grad, torch.tensor(3 * (5.0 + 7.0)))

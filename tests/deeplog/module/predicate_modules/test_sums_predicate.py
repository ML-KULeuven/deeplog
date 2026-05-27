#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import SumsPredicate
from deeplog import SymTensor
from deeplog import reshape


def test_sums_predicate_single_triplet():
    a, b, total = ("A",), ("B",), ("Total",)
    predicate = SumsPredicate([(a, b, total)])

    assignments = torch.tensor([[1.0, 2.0, 3.0], [5.0, 4.0, 9.0], [2.0, 1.0, 0.0]])

    result = predicate(*assignments.unsqueeze(2).unbind(1))
    expected = torch.tensor([[1.0], [1.0], [0.0]], dtype=assignments.dtype)
    torch.testing.assert_close(result, expected)


def test_sums_predicate_multiple_triplets_and_dtype():
    a, b, c, d = ("A",), ("B",), ("C",), ("D",)

    predicate = reshape(
        SumsPredicate([(a, b, c), (b, c, d)]), input=SymTensor([a, b, c, d])
    )

    assignments = torch.tensor(
        [[1, 2, 3, 5], [3, 4, 7, 11], [1, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long
    )

    result = predicate(assignments)
    expected = torch.tensor([[1, 1], [1, 1], [1, 1], [0, 1]], dtype=torch.long)
    torch.testing.assert_close(result, expected)

    input_symbols = list(predicate.get_input_shape())
    assert input_symbols == [a, b, c, d]


def test_sums_predicate_empty_triplets():
    predicate = SumsPredicate([])
    result = predicate(
        torch.zeros(4, 0),
        torch.zeros(4, 0),
        torch.zeros(4, 0),
    )
    assert result.shape == (4, 0)

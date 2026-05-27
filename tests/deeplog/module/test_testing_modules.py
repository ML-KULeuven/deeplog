#  Copyright (c) 2024-2026. KU Leuven
import torch

from ..testing_modules import IndexClassifier


def test_index_classifier_distribution():
    classifier = IndexClassifier(num_classes=3, peak_probability=0.8)
    inputs = torch.tensor([0.0, 2.0, 1.0])

    probs = classifier(inputs)

    expected = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
        ]
    )

    torch.testing.assert_close(probs, expected)


def test_index_classifier_accepts_integer_inputs():
    classifier = IndexClassifier(num_classes=2, peak_probability=0.95)
    inputs = torch.tensor([1, 0], dtype=torch.int64)
    probs = classifier(inputs)

    torch.testing.assert_close(
        probs,
        torch.tensor([[0.05, 0.95], [0.95, 0.05]], dtype=torch.float32),
    )

#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import SymTensor
from deeplog.formula.deeplogmodulefactory.nodes import build_transform


a, b = ("a",), ("b",)


def test_real_to_probability_cast():
    input_shape = SymTensor([a, b])
    module = build_transform(input_shape, "real", "probability")

    assert module.get_input_shape() == SymTensor([a, b])
    expected_output_shape = SymTensor(
        [
            ("_", ("transform", ("probability",), a), ("probability",)),
            ("_", ("transform", ("probability",), b), ("probability",)),
        ]
    )
    assert module.get_output_shape() == expected_output_shape
    assert module.get_structure() == "probability"

    input_tensor = torch.tensor([[-5.0, 3.0], [2.0, 7.0]])
    torch.testing.assert_close(module(input_tensor), torch.sigmoid(input_tensor))

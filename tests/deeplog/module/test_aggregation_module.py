#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import AggregationModule
from deeplog import SymTensor
from deeplog import WrappedModule


a, b, c = ((x,) for x in "abc")


def test_aggregation_module():
    aggregated_module = WrappedModule(
        lambda x: x,
        SymTensor([a, b, c]),
        SymTensor([a, b, c]),
    )
    aggregation_module = AggregationModule(
        aggregated_module, [b], [torch.tensor([0, 1])], "sum"
    )

    assert aggregation_module.get_input_shape() == SymTensor([a, c])
    assert aggregation_module.get_output_shape() == SymTensor(
        [
            ("sum", ("binders", b), a),
            ("sum", ("binders", b), b),
            ("sum", ("binders", b), c),
        ]
    )

    output = aggregation_module(torch.tensor([[0.2, 0.8], [0.3, 0.7]]))
    expected = torch.tensor([[0.4, 1.0, 1.6], [0.6, 1.0, 1.4]])

    torch.testing.assert_close(output, expected)

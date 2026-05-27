#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import SymTensor
from deeplog import WrappedModule


def test_wrapped_module_behaves_like_torch_network():
    class SumNet(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    input_shape = (SymTensor("a"), SymTensor("b"))
    output_shape = SymTensor("sum(a,b)")
    wrapped = WrappedModule(SumNet(), input_shape, output_shape, name="sum_net")

    x = torch.arange(3, dtype=torch.float32).view(3, 1)
    y = torch.arange(3, 6, dtype=torch.float32).view(3, 1)

    out = wrapped(x, y)
    torch.testing.assert_close(out, x + y)
    assert wrapped.get_input_shape() == input_shape
    assert wrapped.get_output_shape() == output_shape

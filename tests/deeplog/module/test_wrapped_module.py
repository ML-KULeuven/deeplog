#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog.shape import ShapeMismatchException


def test_zero_width_input_callable_with_no_args():
    """A WrappedModule whose only input channel is zero-width is callable bare.

    All features were baked away (a fully-constant circuit), so there is nothing
    to pass but the batch size; calling with no args synthesises one row.
    """

    class Const(torch.nn.Module):
        def forward(self, x):  # x: (..., 0) — only carries the batch dim
            return torch.full(x.shape[:-1] + (1,), 0.42)

    wrapped = WrappedModule(
        Const(), SymTensor([]), SymTensor("p"), name="const", vmap=True
    )

    bare = wrapped()  # no args at all
    torch.testing.assert_close(bare, torch.tensor([[0.42]]))
    # The empty (batch, 0) channel still works and carries an explicit batch.
    torch.testing.assert_close(wrapped(torch.zeros(3, 0)), torch.full((3, 1), 0.42))


def test_real_input_still_required():
    """The bare-call shortcut must not mask a genuinely missing input."""
    wrapped = WrappedModule(
        torch.nn.Sigmoid(), SymTensor("x"), SymTensor("x"), name="sig"
    )
    with pytest.raises(ShapeMismatchException):
        wrapped()


def test_name_defaults_without_dunder_name():
    """``name`` is optional: an nn.Module instance has no ``__name__``."""

    def relu(x):
        return torch.relu(x)

    module = WrappedModule(torch.nn.Sigmoid(), SymTensor("x"), SymTensor("x"))
    assert module.name == "Sigmoid"
    # A callable that does carry __name__ keeps using it.
    assert WrappedModule(relu, SymTensor("x"), SymTensor("x")).name == "relu"


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

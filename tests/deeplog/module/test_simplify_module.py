#  Copyright (c) 2024-2026. KU Leuven

import torch

from deeplog import SymTensor
from deeplog import simplify_module

from ._utils import DummyModule


def test_simplify_single_empty_input():
    empty = SymTensor([])
    captured = {}

    def forward(x):
        captured["arg"] = x
        return x

    module = DummyModule(empty, empty, forward)
    simplified = simplify_module(module)

    assert simplified.get_input_shape() == tuple()

    output = simplified()
    assert output.shape == torch.Size([1, 0])
    assert captured["arg"].shape == torch.Size([1, 0])


def test_simplify_mixed_empty_inputs():
    empty = SymTensor([])
    x = SymTensor(["x"])
    captured = {}
    t_x = torch.ones(2, 1)

    def forward(e1, x_arg, e2):
        captured["e1"] = e1
        captured["x"] = x_arg
        captured["e2"] = e2
        return x_arg

    module = DummyModule((empty, x, empty), x, forward)
    simplified = simplify_module(module)

    assert simplified.get_input_shape() == x

    output = simplified(t_x)
    assert output.shape == torch.Size([2, 1])
    assert captured["e1"].shape == torch.Size([2, 0])
    assert captured["e2"].shape == torch.Size([2, 0])
    assert torch.equal(captured["x"], t_x)


def test_simplify_duplicate_inputs():
    x = SymTensor(["x", "x"])
    y = SymTensor(["y"])
    t_x = torch.randn(2, 1)  # Matches deduped public shape
    t_y = torch.randn(2, 1)
    captured = {}

    def forward(a, c):
        captured["a"] = a
        captured["c"] = c
        return torch.cat((a, c), dim=1)

    module = DummyModule((x, y), SymTensor(["x", "x", "y"]), forward)
    simplified = simplify_module(module)

    # x dedupes to a single symbol
    assert simplified.get_input_shape() == (SymTensor(["x"]), y)

    output = simplified(t_x, t_y)
    assert output.shape == torch.Size([2, 3])
    assert torch.equal(captured["c"], t_y)
    # The deduped x should reconstruct both positions
    assert captured["a"].shape == torch.Size([2, 2])
    expanded = torch.cat((t_x, t_x), dim=1)
    assert torch.equal(captured["a"], expanded)


def test_simplify_single_element_tuple():
    x = SymTensor(["x"])
    t_x = torch.ones(1, 1)

    def forward(x_arg):
        return x_arg

    module = DummyModule((x,), x, forward)
    simplified = simplify_module(module)

    assert simplified.get_input_shape() == x
    output = simplified(t_x)
    assert torch.equal(output, t_x)


def test_simplified_repr_uses_original_name():
    empty = SymTensor([])

    def forward(x):
        return x

    module = DummyModule(empty, empty, forward)
    simplified = simplify_module(module)

    repr_str = repr(simplified)
    assert "Sequential" in repr_str or repr_str
    assert "()" in repr_str
